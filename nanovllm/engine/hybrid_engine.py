"""Single-device eager hybrid engine with explicit per-request state ownership."""
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from time import perf_counter

import torch
from transformers import AutoProcessor, AutoTokenizer

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.hybrid_cache import CompressionConfig, HybridCache, StatePool
from nanovllm.models.qwen3_5 import Qwen35Model


@dataclass
class HybridRequest:
    request_id: int
    tokens: list[int]
    prompt_length: int
    params: SamplingParams
    embeddings: torch.Tensor | None
    positions: torch.Tensor
    visual_positions: set
    submitted: float = field(default_factory=perf_counter)
    emitted: list[float] = field(default_factory=list)
    kv_peak: int = 0
    kv_after: int = 0
    memory: dict = field(default_factory=dict)
    compressions: int = 0


class HybridLLM:
    """Dense Qwen3.5 reference-kernel backend independent of Qwen3's NCCL path.

    Round-robin requests have isolated state slots. Device execution is serial;
    fused batching, paged K/V, TP and CUDA graphs are not supported here.
    """
    def __init__(self, model, *, device="cuda", dtype=None, compression=None,
                 max_num_seqs=8, max_num_batched_tokens=16384, max_model_len=32768,
                 tensor_parallel_size=1, enforce_eager=True, **kwargs):
        if kwargs:
            raise TypeError(f"unknown hybrid configuration: {sorted(kwargs)}")
        if tensor_parallel_size != 1:
            raise NotImplementedError("hybrid backend currently supports TP=1 only")
        if not enforce_eager:
            raise NotImplementedError("dynamic hybrid caches require eager execution")
        if min(max_num_seqs, max_num_batched_tokens, max_model_len) < 1:
            raise ValueError("scheduler budgets must be positive")
        dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
        self.runner = Qwen35Model.from_pretrained(model, device, dtype)
        self.model_path = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.processor = AutoProcessor.from_pretrained(model) if self.runner.multimodal else None
        self.compression = compression or CompressionConfig()
        self.max_model_len = min(max_model_len, self.runner.config.max_position_embeddings)
        self.chunk_size = max_num_batched_tokens
        self.pool = StatePool(max_num_seqs, lambda: HybridCache(self.runner.config, self.compression))
        self.pending, self.active = deque(), deque()
        self.requests, self.completed, self.metrics = {}, {}, {}
        self.counter = count()
        self.closed = False

    def _sync(self):
        if self.runner.device.type == "cuda":
            torch.cuda.synchronize(self.runner.device)

    def _encode(self, prompt):
        if isinstance(prompt, dict):
            if "messages" not in prompt:
                raise ValueError("structured input requires messages")
            if self.processor is None:
                raise ValueError("messages/image input requires a multimodal checkpoint")
            return self.processor.apply_chat_template(prompt["messages"], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one token")
        if any(not isinstance(t, int) or t < 0 or t >= self.runner.config.vocab_size for t in ids):
            raise ValueError("invalid prompt token ID")
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def add_request(self, prompt, sampling_params):
        if self.closed:
            raise RuntimeError("engine is closed")
        started = perf_counter()
        inputs = self._encode(prompt)
        length = inputs["input_ids"].shape[1]
        if length + sampling_params.max_tokens > self.max_model_len:
            raise ValueError("prompt + max_tokens exceeds max_model_len")
        ids, embeds, positions, visual = self.runner.prepare(inputs)
        if self.compression.budget and len(visual) + self.compression.window > self.compression.budget:
            raise ValueError("compression budget cannot protect visual tokens and recent window")
        request_id = next(self.counter)
        req = HybridRequest(request_id, ids[0].tolist(), length, sampling_params,
                            embeds, positions, visual, submitted=started)
        self.requests[request_id] = req
        self.pending.append(request_id)
        return request_id

    def is_finished(self):
        return not self.pending and not self.active

    def cancel(self, request_id):
        if request_id in self.active:
            self.active.remove(request_id)
            self.pool.release(request_id)
        elif request_id in self.pending:
            self.pending.remove(request_id)
        else:
            raise KeyError(request_id)
        del self.requests[request_id]

    def preempt(self, request_id):
        """Release all state and recompute token history; lossy caches may differ.

        Recompute is not a snapshot restore. Speculation uses transactions.
        """
        self.active.remove(request_id)
        self.pool.release(request_id)
        self.pending.appendleft(request_id)

    def _positions(self, req, start, end):
        if end <= req.prompt_length:
            return req.positions[..., start:end]
        delta = int(req.positions.max().item()) + 1 - req.prompt_length
        return (torch.arange(start, end, device=self.runner.device) + delta)[None, None].expand(3, 1, -1)

    @torch.inference_mode()
    def step(self):
        while self.pending and self.pool.free:
            rid = self.pending.popleft()
            cache = self.pool.allocate(rid)
            cache.protected_positions = self.requests[rid].visual_positions.copy()
            self.active.append(rid)
        if not self.active:
            return [], 0
        rid = self.active.popleft()
        req, cache = self.requests[rid], self.pool.get(rid)
        start = cache.logical_length
        limit = req.prompt_length if start < req.prompt_length else len(req.tokens)
        end = min(limit, start + self.chunk_size)
        ids = torch.tensor([req.tokens[start:end]], device=self.runner.device)
        embeddings = None if req.embeddings is None or start >= req.prompt_length else req.embeddings[:, start:end]
        try:
            logits = self.runner.forward(ids, cache, embeddings, self._positions(req, start, end))
            req.kv_peak = max(req.kv_peak, cache.memory_bytes()["kv_bytes"])
            cache.compact(force=end == req.prompt_length)
            req.memory = cache.memory_bytes()
            req.kv_after = req.memory["kv_bytes"]
            req.compressions = cache.compressions
            if end < len(req.tokens):
                self.active.append(rid)
                return [], end - start
            if req.params.temperature == 0:
                token = int(logits[0, -1].argmax())
            else:
                probs = (logits[0, -1] / req.params.temperature).softmax(-1)
                token = int(torch.multinomial(probs, 1))
            self._sync()
        except Exception:
            self.pool.release(rid)
            del self.requests[rid]
            raise
        req.tokens.append(token)
        req.emitted.append(perf_counter())
        finished = len(req.emitted) >= req.params.max_tokens or (
            not req.params.ignore_eos and token == self.tokenizer.eos_token_id)
        if finished:
            output = req.tokens[req.prompt_length:]
            self.completed[rid] = output
            self.metrics[rid] = self._metrics(req)
            self.pool.release(rid)
            del self.requests[rid]
            return [(rid, output)], end - start
        self.active.append(rid)
        return [], end - start

    def _metrics(self, req):
        first, last = req.emitted[0], req.emitted[-1]
        n = len(req.emitted)
        return {"request_id": req.request_id, "input_tokens": req.prompt_length,
                "output_tokens": n, "ttft_ms": (first - req.submitted) * 1000,
                "tpot_ms": (last - first) * 1000 / (n - 1) if n > 1 else None,
                "itl_ms": [(b-a)*1000 for a, b in zip(req.emitted, req.emitted[1:])],
                "decode_tokens_per_second": (n-1)/(last-first) if n > 1 else None,
                "kv_peak_bytes": req.kv_peak, "kv_final_bytes": req.kv_after,
                "state_memory": req.memory, "compressions": req.compressions,
                "e2e_ms": (last - req.submitted) * 1000}

    def generate(self, prompts, sampling_params, use_tqdm=False):
        if not self.is_finished():
            raise RuntimeError("drain outstanding requests before generate")
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError("one SamplingParams is required for each prompt")
        ids = []
        try:
            for prompt, param in zip(prompts, params):
                ids.append(self.add_request(prompt, param))
            while not self.is_finished():
                self.step()
        except Exception:
            for rid in list(self.requests):
                self.cancel(rid)
            raise
        return [{"text": self.tokenizer.decode(self.completed[rid], skip_special_tokens=True),
                 "token_ids": self.completed.pop(rid), "metrics": self.metrics.pop(rid)} for rid in ids]

    @torch.inference_mode()
    def generate_mtp(self, prompt, sampling_params, depth=3):
        """Experimental single-request MTP with serial verification and replay.

        Uses actual checkpoint MTP weights. Draft history is rebuilt each round
        for clarity; this is not a speed optimization. Compression is frozen
        during verification and committed at round boundaries. With compression,
        correctness is relative to that target schedule, not an uncompressed model.
        """
        from nanovllm.models.qwen3_5_mtp import MTPDraft
        from nanovllm.engine.speculation import GreedySpeculator
        if sampling_params.temperature != 0:
            raise NotImplementedError("MTP prototype requires greedy sampling")
        if not self.is_finished():
            raise RuntimeError("drain outstanding requests before MTP generation")
        if depth < 1:
            raise ValueError("MTP depth must be positive")
        if not hasattr(self, "mtp_draft"):
            self.mtp_draft = MTPDraft.from_checkpoint(self.runner, self.model_path)
        rid = self.add_request(prompt, sampling_params)
        self.pending.remove(rid)
        req = self.requests[rid]
        cache = self.pool.allocate(rid)
        cache.protected_positions = req.visual_positions.copy()
        rounds, proposed, accepted_drafts, bursts = 0, 0, 0, []
        try:
            ids = torch.tensor([req.tokens], device=self.runner.device)
            logits, hidden = self.runner.forward(ids, cache, req.embeddings, req.positions, return_hidden=True)
            while len(req.emitted) < req.params.max_tokens:
                # Previous round's bonus was emitted but has not been consumed.
                if cache.logical_length < len(req.tokens):
                    offset = cache.logical_length
                    logits, last_hidden = self.runner.forward(
                        torch.tensor([[req.tokens[-1]]], device=self.runner.device), cache,
                        position_ids=self._positions(req, offset, offset+1), return_hidden=True)
                    hidden = torch.cat((hidden, last_hidden), dim=1)
                req.kv_peak = max(req.kv_peak, cache.memory_bytes()["kv_bytes"])
                cache.compact(force=rounds == 0)
                consumed = cache.logical_length
                all_ids = torch.tensor([req.tokens[:consumed]], device=self.runner.device)
                positions = torch.cat((req.positions, self._positions(req, req.prompt_length, consumed)), -1) if consumed > req.prompt_length else req.positions
                draft_cache = self.mtp_draft.prime(all_ids, hidden, positions)
                first = int(logits.argmax())
                candidates = self.mtp_draft.propose(first, hidden, draft_cache,
                    self._positions(req, consumed, consumed+1), depth)
                captured = {}
                def consume(token, state):
                    offset = state.logical_length
                    scores, token_hidden = self.runner.forward(
                        torch.tensor([[token]], device=self.runner.device), state,
                        position_ids=self._positions(req, offset, offset+1), return_hidden=True)
                    captured[offset] = (scores, token_hidden)
                    return scores
                remaining = req.params.max_tokens-len(req.emitted)
                result = GreedySpeculator().verify(cache, logits, candidates, consume,
                    max_tokens=remaining, eos_token_id=None if req.params.ignore_eos else self.tokenizer.eos_token_id)
                for offset in range(consumed, cache.logical_length):
                    logits, token_hidden = captured[offset]
                    hidden = torch.cat((hidden, token_hidden), dim=1)
                new_tokens = result.accepted + [result.next_token]
                req.tokens.extend(new_tokens)
                self._sync()
                req.emitted.extend([perf_counter()]*len(new_tokens))
                bursts.append(len(new_tokens))
                proposed += max(0, result.proposed-1)  # first token is already target-selected
                accepted_drafts += max(0, len(result.accepted)-1)
                rounds += 1
                req.kv_peak = max(req.kv_peak, cache.memory_bytes()["kv_bytes"])
                req.memory = cache.memory_bytes()
                req.kv_after = req.memory["kv_bytes"]
                req.compressions = cache.compressions
                if not req.params.ignore_eos and result.next_token == self.tokenizer.eos_token_id:
                    break
            tokens = req.tokens[req.prompt_length:]
            return {"text": self.tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens,
                    "metrics": self._metrics(req), "mtp": {"rounds": rounds, "draft_tokens": proposed,
                    "accepted_draft_tokens": accepted_drafts, "bursts": bursts,
                    "serial_verification": True, "speedup_claim": False}}
        finally:
            self.pool.release(rid)
            del self.requests[rid]

    def exit(self):
        if self.closed:
            return
        for rid in list(self.requests):
            self.cancel(rid)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exit()
