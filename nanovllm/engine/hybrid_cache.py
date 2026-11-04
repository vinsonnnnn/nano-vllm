"""Request-owned Qwen3.5 state, compact attention storage and transactions.

The reference GDN kernels own the convolution/recurrent tensors. Only full
attention K/V is compressed; logical positions and GDN history never shrink.
This eager backend deliberately does not share mutable prefix caches.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache


@dataclass(frozen=True)
class CompressionConfig:
    budget: int = 0
    window: int = 32
    interval: int = 64
    pool: int = 5
    min_reclaim: int = 1

    def __post_init__(self):
        if self.budget < 0 or min(self.window, self.interval, self.min_reclaim) < 1:
            raise ValueError("invalid compression budget/window/interval/min_reclaim")
        if self.budget and self.budget <= self.window:
            raise ValueError("budget must exceed the recent observation window")
        if self.pool < 1 or self.pool % 2 == 0:
            raise ValueError("pool must be a positive odd number")


def select_positions(scores, length, config, protected=()):
    """One chronological selection shared by heads, independently per layer."""
    if not config.budget or length <= config.budget:
        return torch.arange(length, device=scores.device)
    recent = torch.arange(max(0, length - config.window), length, device=scores.device)
    mandatory = torch.unique(torch.cat((recent, torch.as_tensor(
        protected, device=scores.device, dtype=torch.long))))
    if mandatory.numel() > config.budget:
        raise ValueError("protected visual tokens exceed compression budget")
    pooled = F.max_pool1d(scores[None, None].float(), config.pool, stride=1,
                          padding=config.pool // 2).flatten()
    pooled[mandatory] = -torch.inf
    # Stable tie breaking makes tests and repeated greedy runs reproducible.
    ranked = torch.argsort(pooled, descending=True, stable=True)
    keep = torch.cat((mandatory, ranked[:config.budget - mandatory.numel()]))
    return keep.sort().values


class HybridCache(Qwen3_5DynamicCache):
    def __init__(self, config, compression=None):
        super().__init__(config)
        self.compression = compression or CompressionConfig()
        self.logical_length = 0
        self.positions = [None] * config.num_hidden_layers
        self.queries = [None] * config.num_hidden_layers
        self.query_positions = [None] * config.num_hidden_layers
        self.protected_positions = set()
        self.last_compression = 0
        self.compressions = 0
        self.frozen = False  # speculation suppresses lossy mutations until commit

    def get_seq_length(self, layer_idx=0):
        return self.logical_length

    def get_mask_sizes(self, cache_position, layer_idx):
        # Physical mask sizes must not follow the logical RoPE coordinates.
        idx = layer_idx if layer_idx in self.transformer_layers else self.transformer_layers[0]
        k = self.key_cache[idx]
        return (0 if k is None else k.shape[-2]) + cache_position.numel(), 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pos = cache_kwargs["cache_position"]
        old = self.positions[layer_idx]
        self.positions[layer_idx] = pos.clone() if old is None else torch.cat((old, pos))
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def observe(self, layer_idx, queries, positions):
        if not self.compression.budget:
            return
        old_q, old_p = self.queries[layer_idx], self.query_positions[layer_idx]
        q = queries.detach() if old_q is None else torch.cat((old_q, queries.detach()), dim=-2)
        p = positions if old_p is None else torch.cat((old_p, positions))
        self.queries[layer_idx] = q[..., -self.compression.window:, :].clone()
        self.query_positions[layer_idx] = p[-self.compression.window:].clone()

    def compact(self, force=False):
        cfg = self.compression
        if self.frozen or not cfg.budget:
            return False
        if not force and self.logical_length - self.last_compression < cfg.interval:
            return False
        replacements = []
        for idx in self.transformer_layers:
            key, value = self.key_cache[idx], self.value_cache[idx]
            if key is None or key.shape[-2] - cfg.budget < cfg.min_reclaim:
                continue
            query = self.queries[idx]
            groups = query.shape[1] // key.shape[1]
            # O(window * retained_length), never a full prompt attention matrix.
            expanded = key.repeat_interleave(groups, dim=1)
            logits = (query.float() @ expanded.float().transpose(-1, -2)) / math.sqrt(key.shape[-1])
            causal = self.positions[idx][None, :] <= self.query_positions[idx][:, None]
            weights = logits.masked_fill(~causal, -torch.inf).softmax(dim=-1)
            scores = weights.mean(dim=(0, 1, 2))
            protected = [i for i, p in enumerate(self.positions[idx].tolist())
                         if p in self.protected_positions]
            keep = select_positions(scores, key.shape[-2], cfg, protected)
            replacements.append((idx, key.index_select(-2, keep).contiguous(),
                                 value.index_select(-2, keep).contiguous(),
                                 self.positions[idx].index_select(0, keep)))
        # Allocate/gather every layer before publishing: exceptions leave old K/V intact.
        for idx, key, value, positions in replacements:
            self.key_cache[idx], self.value_cache[idx], self.positions[idx] = key, value, positions
        self.last_compression = self.logical_length
        self.compressions += bool(replacements)
        return bool(replacements)

    def memory_bytes(self):
        def size(items):
            return sum(t.numel() * t.element_size() for t in items if t is not None)
        return {"kv_bytes": size(self.key_cache) + size(self.value_cache),
                "recurrent_bytes": size(self.recurrent_states),
                "conv_bytes": size(self.conv_states),
                "observation_bytes": size(self.queries) + size(self.query_positions),
                "position_bytes": size(self.positions)}

    def snapshot(self):
        return deepcopy(self)

    def restore(self, snapshot):
        # Clone again so repeated rollback does not mutate the saved checkpoint.
        self.__dict__ = deepcopy(snapshot.__dict__)


class StatePool:
    """Slots belong to request IDs, never to a transient batch row."""
    def __init__(self, capacity, factory):
        if capacity < 1:
            raise ValueError("state capacity must be positive")
        self.capacity, self.factory = capacity, factory
        self.slots, self.owners = {}, {}
        self.free = list(reversed(range(capacity)))

    def allocate(self, request_id):
        if request_id in self.owners:
            raise ValueError("request already has a state slot")
        if not self.free:
            raise MemoryError("no free hybrid state slot")
        state = self.factory()  # failure must not consume a slot
        slot = self.free.pop()
        self.owners[request_id] = slot
        self.slots[slot] = state
        return state

    def get(self, request_id):
        return self.slots[self.owners[request_id]]

    def release(self, request_id):
        slot = self.owners.pop(request_id)
        del self.slots[slot]  # fresh tensors on reuse, no cross-request history
        self.free.append(slot)
