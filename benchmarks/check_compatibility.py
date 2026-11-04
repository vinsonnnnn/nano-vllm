"""Real-checkpoint integration evidence, run separately from latency benchmarks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from nanovllm import HybridLLM, SamplingParams
from nanovllm.engine.hybrid_cache import HybridCache, CompressionConfig
from nanovllm.engine.speculation import GreedySpeculator
from nanovllm.models.qwen3_5_mtp import MTPDraft


def compare_caches(a, b):
    assert a.logical_length == b.logical_length
    for name in ["key_cache", "value_cache", "conv_states", "recurrent_states"]:
        for x, y in zip(getattr(a, name), getattr(b, name)):
            if x is not None:
                torch.testing.assert_close(x, y, atol=.002, rtol=.02)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="reports/qwen35_rtx4090/compatibility.json")
    args = p.parse_args()
    torch.manual_seed(20261104)
    torch.set_num_threads(4)
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = "The quiet library contains maps of France and books on astronomy. " * 6
    ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(args.model,
        dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    output = reference(ids, use_cache=True, logits_to_keep=1)
    expected = [output.logits.detach().float().cpu()]
    history = output.past_key_values
    for t in [101, 102, 103]:
        output = reference(torch.tensor([[t]], device="cuda"), past_key_values=history,
                           use_cache=True, logits_to_keep=1)
        expected.append(output.logits.detach().float().cpu())
    del reference, output, history
    torch.cuda.empty_cache()
    llm = HybridLLM(args.model, max_num_seqs=2, max_model_len=4096)
    model = llm.runner
    cache = HybridCache(model.config)
    actual = [model.forward(ids, cache).cpu()]
    for t in [101, 102, 103]:
        actual.append(model.forward(torch.tensor([[t]], device="cuda"), cache).cpu())
    errors = [(a-b).abs().max().item() for a, b in zip(actual, expected)]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=.125, rtol=.02)
        assert int(a.argmax()) == int(b.argmax())
    results["official_bf16_alignment"] = {"max_logit_abs_error_by_step": errors,
                                          "argmax_equal": True, "teacher_forcing_steps": 4}

    full, chunk = HybridCache(model.config), HybridCache(model.config)
    expected_last = model.forward(ids, full)
    model.forward(ids[:, :11], chunk)
    chunk_last = model.forward(ids[:, 11:], chunk)[:, -1:]
    # BF16 chunked/recurrent arithmetic can differ from the chunkwise reference.
    assert int(expected_last.argmax()) == int(chunk_last.argmax())
    results["chunked_prefill"] = {"argmax_equal": True,
        "max_logit_abs_error": (expected_last-chunk_last).abs().max().item(),
        "logical_length": chunk.logical_length, "chunks": [11, ids.shape[1]-11]}

    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    prompts = ["The capital of France is", "One plus one equals"]
    isolated = [llm.generate([s], params)[0]["token_ids"] for s in prompts]
    # Round-robin batches reorder active rows every step; ownership must remain stable.
    together = [r["token_ids"] for r in llm.generate(prompts, params)]
    assert together == isolated
    rid = llm.add_request(prompts[0], params)
    llm.step()
    llm.preempt(rid)
    while not llm.is_finished():
        llm.step()
    assert llm.completed.pop(rid) == isolated[0]
    llm.metrics.pop(rid)
    rid = llm.add_request("Cancel and reuse this state slot", params)
    llm.step()
    llm.cancel(rid)
    assert not llm.pool.owners
    assert llm.generate([prompts[0]], params)[0]["token_ids"] == isolated[0]
    results["request_lifecycle"] = {"interleaving_equal": True, "preempt_recompute_equal": True,
                                    "cancel_reuse_equal": True, "slots_released": True}

    # Keep original absolute positions, with recurrent and convolution states unchanged.
    compressed = HybridCache(model.config, CompressionConfig(budget=48, window=8, interval=8))
    logits, target_hidden = model.forward(ids, compressed, return_hidden=True)
    before = compressed.snapshot()
    assert compressed.compact(force=True)
    for name in ["conv_states", "recurrent_states"]:
        for a, b in zip(getattr(compressed, name), getattr(before, name)):
            if a is not None:
                assert torch.equal(a, b)
    results["compression_hybrid_compatibility"] = {"gdn_and_conv_bitwise_unchanged": True,
        "logical_length": compressed.logical_length, "retained_per_full_layer": [
            compressed.key_cache[i].shape[-2] for i in compressed.transformer_layers],
        "before": before.memory_bytes(), "after": compressed.memory_bytes()}

    draft = MTPDraft.from_checkpoint(model, args.model)
    positions = torch.arange(ids.shape[1], device="cuda")[None, None].expand(3, 1, -1)
    draft_cache = draft.prime(ids, target_hidden, positions)
    candidates = draft.propose(int(logits.argmax()), target_hidden, draft_cache,
                               positions[..., -1:] + 1, depth=3)
    snapshot = compressed.snapshot()
    def consume(token, state):
        return model.forward(torch.tensor([[token]], device="cuda"), state)
    verification = GreedySpeculator().verify(compressed, logits, candidates, consume)
    oracle = snapshot.snapshot()
    next_logits = logits
    for token in verification.accepted:
        next_logits = consume(token, oracle)
    assert verification.next_token == int(next_logits.argmax())
    compare_caches(compressed, oracle)
    # Force a rejection after one accepted position, independent of model acceptance.
    compressed.restore(snapshot)
    first = int(logits.argmax())
    trial = snapshot.snapshot()
    second = int(consume(first, trial).argmax())
    forced = GreedySpeculator().verify(compressed, logits, [first, (second+1)%model.config.vocab_size], consume)
    assert forced.accepted == [first] and forced.next_token == second
    compare_caches(compressed, trial)
    results["mtp_prototype"] = {"loaded_mtp_tensor_count": len(draft.loaded_keys),
        "candidates": candidates, "accepted": verification.accepted,
        "next_token": verification.next_token, "matches_fixed_compressed_target": True,
        "forced_rejection_restores_gdn_conv_kv": True, "draft_cache_unmodified": draft_cache.logical_length == ids.shape[1],
        "scope": "one-round real MTP draft + serial greedy target verification; no speedup claim or generate integration"}

    llm.compression = CompressionConfig()
    mtp_params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
    target_output = llm.generate([prompt], mtp_params)[0]
    mtp_output = llm.generate_mtp(prompt, mtp_params, depth=3)
    assert target_output["token_ids"] == mtp_output["token_ids"]
    results["mtp_generation"] = {"greedy_equal_without_compression": True,
        "tokens": mtp_output["token_ids"], **mtp_output["mtp"]}
    llm.compression = CompressionConfig(budget=48, window=8, interval=8)
    compressed_mtp = llm.generate_mtp(prompt, mtp_params, depth=3)
    assert len(compressed_mtp["token_ids"]) == 12 and not llm.pool.owners
    results["mtp_compressed_generation"] = {"tokens": compressed_mtp["token_ids"],
        "compression_count": compressed_mtp["metrics"]["compressions"],
        "slots_released": True, **compressed_mtp["mtp"]}

    # Exercise the official processor, visual injection, MRoPE, then compressed decode.
    image = Image.new("RGB", (64, 64), "red")
    prompt = {"messages": [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": ("This is a simple image recognition task. "*20 + "What is the main color? Answer with one word.")}]}]}
    llm.compression = CompressionConfig(budget=128, window=8, interval=2)
    encoded = llm._encode(prompt)
    image_ids, embeds, image_positions, protected = model.prepare(encoded)
    mm_cache = HybridCache(model.config, llm.compression)
    mm_cache.protected_positions = protected
    mm_logits = model.forward(image_ids, mm_cache, embeds, image_positions)
    mm_cache.compact(force=True)
    for idx in mm_cache.transformer_layers:
        assert protected.issubset(set(mm_cache.positions[idx].tolist()))
    response = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8))[0]
    assert "red" in response["text"].lower()
    results["multimodal"] = {"text": response["text"], "visual_token_count": len(protected),
        "all_visual_positions_retained": True, "position_shape": list(image_positions.shape),
        "finite_logits": bool(torch.isfinite(mm_logits).all()), "metrics": response["metrics"]}
    llm.exit()
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
