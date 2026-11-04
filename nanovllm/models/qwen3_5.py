"""Portable dense Qwen3.5 adapter using pinned Transformers reference kernels.

No Transformers.generate call: scheduling, consumed positions, K/V compaction
and state transactions belong to nano-vLLM. Reference vision, MRoPE and GDN
are reused rather than approximated. Full attention reads compact physical K/V.
"""
from types import MethodType
import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def compact_attention(self, hidden_states, position_embeddings, attention_mask=None,
                      past_key_values=None, cache_position=None, **kwargs):
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, self.head_dim)
    q, gate = self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2).chunk(2, -1)
    q = self.q_norm(q.reshape(head_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k, v = past_key_values.update(k, v, self.layer_idx, {"cache_position": cache_position})
    past_key_values.observe(self.layer_idx, q, cache_position)
    # Original absolute positions are preserved after gathering K/V.
    first_prefill = q.shape[-2] == k.shape[-2]
    # Match the reference SDPA dispatch for contiguous prefill and single-token
    # decode. Supplying an unnecessary all-true mask can select a different
    # BF16 kernel and introduce avoidable rounding differences.
    mask = None
    if not first_prefill and q.shape[-2] > 1:
        mask = past_key_values.positions[self.layer_idx][None, :] <= cache_position[:, None]
    if q.shape[-2] > 8192:
        # On Windows, native GQA may select the quadratic-memory math backend.
        # Explicit head expansion permits memory-efficient SDPA kernels instead.
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            is_causal=first_prefill and q.shape[-2] > 1, dropout_p=0.0, scale=self.scaling)
        out = out.transpose(1, 2)
    else:
        out, _ = sdpa_attention_forward(self, q, k, v, mask, dropout=0.0,
                                        scaling=self.scaling, is_causal=first_prefill)
    out = out.reshape(*shape, -1)
    return self.o_proj(out * gate.reshape(*shape, -1).sigmoid()), None


class Qwen35Model:
    def __init__(self, model):
        self.model = model.eval()
        self.multimodal = hasattr(model.model, "language_model")
        self.text = model.model.language_model if self.multimodal else model.model
        self.config = self.text.config
        if "linear_attention" not in self.config.layer_types or "full_attention" not in self.config.layer_types:
            raise ValueError("expected a dense hybrid Qwen3.5 checkpoint")
        for layer in self.text.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.forward = MethodType(compact_attention, layer.self_attn)

    @classmethod
    def from_pretrained(cls, path, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path)
        model_cls = Qwen3_5ForCausalLM if cfg.model_type == "qwen3_5_text" else Qwen3_5ForConditionalGeneration
        if getattr(cfg, "quantization_config", None):
            from nanovllm.utils.fp8 import load_fp8_model
            model = load_fp8_model(model_cls, cfg, path, device, dtype)
        else:
            model = model_cls.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device)
        return cls(model)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def prepare(self, inputs):
        """Encode images once; return embeddings and request-owned 3-axis positions."""
        ids = inputs["input_ids"].to(self.device)
        embeds = self.text.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=self.device)[None, None].expand(3, 1, -1)
        visual_positions = set()
        if inputs.get("pixel_values") is not None:
            if not self.multimodal:
                raise ValueError("text-only checkpoint cannot encode images")
            mm = self.model.model
            grid = inputs["image_grid_thw"].to(self.device)
            features = mm.get_image_features(inputs["pixel_values"].to(self.device, embeds.dtype), grid,
                                             return_dict=True).pooler_output
            features = torch.cat(features, dim=0).to(embeds)
            mask, _ = mm.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
            embeds = embeds.masked_scatter(mask, features)
            mm_types = inputs.get("mm_token_type_ids")
            if mm_types is None:
                mm_types = (ids == self.model.config.image_token_id).long()
            positions, _ = mm.get_rope_index(ids, mm_token_type_ids=mm_types.to(self.device), image_grid_thw=grid,
                                             attention_mask=torch.ones_like(ids))
            visual_positions = set((ids[0] == self.model.config.image_token_id).nonzero().flatten().tolist())
        return ids, embeds, positions, visual_positions

    @torch.inference_mode()
    def forward(self, token_ids, cache, embeddings=None, position_ids=None, return_hidden=False):
        length = token_ids.shape[1]
        # The 5.3.0 reference multi-token GDN path starts from initial_state=None.
        # Continue existing states token by token, including chunked prefill and
        # speculative replay, rather than silently resetting recurrent history.
        if cache.has_previous_state and length > 1:
            results = []
            for i in range(length):
                results.append(self.forward(token_ids[:, i:i+1], cache,
                    None if embeddings is None else embeddings[:, i:i+1],
                    None if position_ids is None else position_ids[..., i:i+1], return_hidden))
            if return_hidden:
                return torch.cat([r[0] for r in results], 1), torch.cat([r[1] for r in results], 1)
            return torch.cat(results, dim=1)
        hidden = self.text.embed_tokens(token_ids) if embeddings is None else embeddings
        absolute = torch.arange(cache.logical_length, cache.logical_length + length, device=self.device)
        if position_ids is None:
            position_ids = absolute[None, None].expand(3, 1, -1)
        rope = self.text.rotary_emb(hidden, position_ids)
        for layer in self.text.layers:
            hidden = layer(hidden, position_embeddings=rope, past_key_values=cache,
                           cache_position=absolute, attention_mask=None)
        hidden = self.text.norm(hidden)
        cache.logical_length += length
        # Prefill needs only the last logits, saving a prompt*vocabulary allocation.
        logits = self.model.lm_head(hidden[:, -1:]).float()
        return (logits, hidden) if return_hidden else logits
