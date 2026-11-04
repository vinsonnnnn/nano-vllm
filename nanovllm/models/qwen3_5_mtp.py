"""Single-layer dense Qwen3.5 MTP drafter, eager experimental prototype.

Architecture contract follows vLLM's Qwen3_5MultiTokenPredictor: normalized
embedding + normalized previous hidden -> concatenation -> fc -> full-attention
decoder -> norm -> shared LM head. This module loads actual mtp.* weights.
It is intentionally separate from the production generate loop: verification is
serial and snapshot-heavy and therefore does not promise decoding acceleration.
"""
from pathlib import Path
from types import MethodType
import torch
from torch import nn
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

from nanovllm.engine.hybrid_cache import HybridCache
from nanovllm.models.qwen3_5 import compact_attention


class MTPDraft(nn.Module):
    def __init__(self, target):
        super().__init__()
        object.__setattr__(self, "target", target)  # shared embedding/LM head, not copied
        cfg = target.config
        if getattr(cfg, "mtp_num_hidden_layers", 1) != 1:
            raise NotImplementedError("MTP prototype supports one trained prediction layer")
        if getattr(cfg, "mtp_use_dedicated_embeddings", False):
            raise NotImplementedError("dedicated MTP embedding checkpoints are unsupported")
        idx = cfg.layer_types.index("full_attention")
        self.fc = nn.Linear(2*cfg.hidden_size, cfg.hidden_size, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, idx)])
        self.layers[0].self_attn.forward = MethodType(compact_attention, self.layers[0].self_attn)
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.to(device=target.device, dtype=next(target.model.parameters()).dtype)
        self.eval()

    @classmethod
    def from_checkpoint(cls, target, path):
        draft = cls(target)
        state = {}
        for file in sorted(Path(path).glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith("mtp."):
                        tensor = handle.get_tensor(key)
                        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                            raise NotImplementedError("MTP prototype requires unquantized MTP weights")
                        state[key.removeprefix("mtp.")] = tensor
        draft.load_state_dict(state, strict=True)
        draft.loaded_keys = sorted(state)
        return draft

    @torch.inference_mode()
    def forward(self, tokens, previous_hidden, cache, positions, absolute):
        embeds = self.target.text.embed_tokens(tokens)
        hidden = self.fc(torch.cat((self.pre_fc_norm_embedding(embeds),
                                   self.pre_fc_norm_hidden(previous_hidden)), dim=-1))
        rope = self.target.text.rotary_emb(hidden, positions)
        hidden = self.layers[0](hidden, position_embeddings=rope, past_key_values=cache,
                                cache_position=absolute, attention_mask=None)
        hidden = self.norm(hidden)
        cache.logical_length = int(absolute[-1]) + 1
        return self.target.model.lm_head(hidden[:, -1:]).float(), hidden

    @torch.inference_mode()
    def prime(self, tokens, target_hidden, positions):
        cache = HybridCache(self.target.config)
        if tokens.shape[1] > 1:
            self.forward(tokens[:, 1:], target_hidden[:, :-1], cache, positions[..., 1:],
                         torch.arange(1, tokens.shape[1], device=tokens.device))
        cache.logical_length = tokens.shape[1]
        return cache

    @torch.inference_mode()
    def propose(self, first_token, target_hidden, cache, positions, depth=3):
        if depth < 1:
            raise ValueError("draft depth must be positive")
        # Draft mutations remain private. The caller can discard an entire branch.
        trial = cache.snapshot()
        candidates = [first_token]
        hidden = target_hidden[:, -1:]
        start = cache.logical_length
        for i in range(depth-1):
            ids = torch.tensor([[candidates[-1]]], device=self.target.device)
            logits, hidden = self.forward(ids, hidden, trial, positions + i,
                torch.tensor([start+i], device=self.target.device))
            candidates.append(int(logits.argmax()))
        return candidates
