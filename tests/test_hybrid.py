from copy import deepcopy
import random

import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

from nanovllm.engine.hybrid_cache import HybridCache, CompressionConfig, StatePool, select_positions
from nanovllm.models.qwen3_5 import Qwen35Model
from nanovllm.engine.speculation import GreedySpeculator
from nanovllm.utils.fp8 import dequantize_block_fp8

torch.set_num_threads(1)


@pytest.fixture
def pair():
    torch.manual_seed(42)
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [2, 1, 1]})
    cfg._attn_implementation = "sdpa"
    reference = Qwen3_5ForCausalLM(cfg).eval()
    return reference, Qwen35Model(deepcopy(reference)), cfg


@torch.inference_mode()
def test_reference_prefill_and_continuous_decode(pair):
    ref, model, cfg = pair
    cache = HybridCache(cfg)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    result = ref(tokens, use_cache=True)
    actual = model.forward(tokens, cache)
    torch.testing.assert_close(actual, result.logits[:, -1:], atol=1e-6, rtol=1e-5)
    ref_cache = result.past_key_values
    for token in [7, 9, 11, 13]:
        ids = torch.tensor([[token]])
        expected = ref(ids, past_key_values=ref_cache, use_cache=True)
        torch.testing.assert_close(model.forward(ids, cache), expected.logits, atol=1e-6, rtol=1e-5)


@torch.inference_mode()
def test_chunked_prefill_and_recompute(pair):
    _, model, cfg = pair
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    full, chunk = HybridCache(cfg), HybridCache(cfg)
    expected = model.forward(ids, full)
    model.forward(ids[:, :3], chunk)
    actual = model.forward(ids[:, 3:], chunk)[:, -1:]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(full.recurrent_states, chunk.recurrent_states):
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_compaction_preserves_gdn_positions_and_reclaims_storage(pair):
    _, model, cfg = pair
    cache = HybridCache(cfg, CompressionConfig(budget=6, window=2, interval=2))
    model.forward(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), cache)
    before = cache.snapshot()
    assert cache.compact(force=True)
    assert cache.logical_length == 10
    for idx in cache.transformer_layers:
        assert cache.key_cache[idx].shape[-2] == 6
        assert cache.key_cache[idx].untyped_storage().nbytes() == cache.key_cache[idx].numel()*4
        assert cache.positions[idx][-2:].tolist() == [8, 9]
        for name in ["conv_states", "recurrent_states"]:
            for a, b in zip(getattr(cache, name), getattr(before, name)):
                if a is not None:
                    assert torch.equal(a, b)
    out = model.forward(torch.tensor([[11]]), cache)
    assert torch.isfinite(out).all()
    assert cache.logical_length == 11
    assert cache.positions[1][-1] == 10
    assert cache.key_cache[1].shape[-2] == 7


@torch.inference_mode()
@pytest.mark.parametrize("reject_at", [0, 1, 2, 3])
def test_speculation_snapshot_and_replay(pair, reject_at):
    _, model, cfg = pair
    cache = HybridCache(cfg, CompressionConfig(budget=5, window=2, interval=1))
    logits = model.forward(torch.tensor([[1, 2, 3, 4, 5, 6]]), cache)
    cache.compact(force=True)
    snapshot = cache.snapshot()
    oracle = cache.snapshot()
    candidates, running = [], logits
    for _ in range(3):
        token = int(running.argmax())
        candidates.append(token)
        running = model.forward(torch.tensor([[token]]), oracle)
    if reject_at < 3:
        candidates[reject_at] = (candidates[reject_at] + 1) % cfg.vocab_size
    def consume(token, state):
        result = model.forward(torch.tensor([[token]]), state)
        assert not state.compact(force=True)
        return result
    result = GreedySpeculator().verify(cache, logits, candidates, consume)
    assert len(result.accepted) == reject_at
    expected = snapshot.snapshot()
    next_logits = logits
    for token in result.accepted:
        next_logits = model.forward(torch.tensor([[token]]), expected)
    assert result.next_token == int(next_logits.argmax())
    assert cache.logical_length == expected.logical_length
    for name in ["key_cache", "value_cache", "conv_states", "recurrent_states"]:
        for a, b in zip(getattr(cache, name), getattr(expected, name)):
            if a is not None:
                torch.testing.assert_close(a, b)


def test_random_slot_reuse_and_ownership(pair):
    _, _, cfg = pair
    pool = StatePool(3, lambda: HybridCache(cfg))
    active = set()
    rng = random.Random(13)
    for rid in range(100):
        if len(active) == 3 or active and rng.random() < .5:
            victim = rng.choice(sorted(active))
            pool.release(victim)
            active.remove(victim)
        state = pool.allocate(rid)
        assert state.logical_length == 0 and all(x is None for x in state.conv_states)
        state.logical_length = rid + 1
        active.add(rid)
        assert len(set(pool.owners.values())) == len(active)
        for owner in active:
            assert pool.get(owner).logical_length == owner + 1


def test_protected_recent_selection_and_validation():
    cfg = CompressionConfig(budget=5, window=2, pool=1)
    keep = select_positions(torch.arange(10).float(), 10, cfg, [0])
    assert keep.tolist() == [0, 6, 7, 8, 9]
    with pytest.raises(ValueError):
        select_positions(torch.ones(10), 10, cfg, list(range(6)))
    with pytest.raises(ValueError):
        CompressionConfig(budget=2, window=2)


@pytest.mark.parametrize("dim", [0, 1])
def test_fp8_scales_and_tp_shards(dim):
    w = torch.arange(32).reshape(4, 8).to(torch.float8_e4m3fn)
    s = torch.tensor([[.5, 2., 3., 4.], [1., 5., 6., 7.]])
    full = dequantize_block_fp8(w, s, [2, 2], dtype=torch.float32)
    expected = w.float()*s.repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert torch.equal(full, expected)
    shards = [dequantize_block_fp8(w, s, [2, 2], shard_dim=dim, rank=r,
                                  world_size=2, dtype=torch.float32) for r in range(2)]
    assert torch.equal(torch.cat(shards, dim=dim), full)


def test_greedy_and_random_sampler_preserve_logits():
    from nanovllm.layers.sampler import Sampler
    sampler = Sampler()
    logits = torch.tensor([[1., 3., 2.], [-1., 2., 0.]])
    original = logits.clone()
    # Exercise the function without invoking a platform-specific compiler toolchain.
    result = sampler.forward.__wrapped__(sampler, logits, torch.tensor([0., 1.]))
    assert result[0].item() == 1
    assert 0 <= result[1].item() < 3
    assert torch.equal(logits, original)


def test_unsupported_hybrid_optimizations_fail_before_loading():
    from nanovllm.engine.hybrid_engine import HybridLLM
    with pytest.raises(NotImplementedError, match="TP=1"):
        HybridLLM("not-a-model", tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="eager"):
        HybridLLM("not-a-model", enforce_eager=False)


@torch.inference_mode()
def test_long_prefill_selects_expanded_head_sdpa(pair, monkeypatch):
    _, model, cfg = pair
    layer = model.text.layers[1].self_attn
    seen = {}
    def sdpa(q, k, v, **kwargs):
        seen["heads"] = (q.shape[1], k.shape[1], v.shape[1])
        seen["causal"] = kwargs["is_causal"]
        return torch.zeros_like(q)
    monkeypatch.setattr("nanovllm.models.qwen3_5.F.scaled_dot_product_attention", sdpa)
    hidden = torch.zeros(1, 8193, cfg.hidden_size)
    positions = torch.arange(8193)
    rope = model.text.rotary_emb(hidden, positions[None, None].expand(3, 1, -1))
    layer(hidden, rope, past_key_values=HybridCache(cfg), cache_position=positions)
    assert seen == {"heads": (2, 2, 2), "causal": True}


@torch.inference_mode()
def test_transaction_exception_restores_every_state(pair):
    _, model, cfg = pair
    cache = HybridCache(cfg)
    logits = model.forward(torch.tensor([[1, 2, 3]]), cache)
    snapshot = cache.snapshot()
    def failure(token, state):
        model.forward(torch.tensor([[token]]), state)
        raise RuntimeError("injected failure")
    with pytest.raises(RuntimeError):
        GreedySpeculator().verify(cache, logits, [int(logits.argmax())], failure)
    assert cache.logical_length == snapshot.logical_length
    for a, b in zip(cache.recurrent_states, snapshot.recurrent_states):
        if a is not None:
            assert torch.equal(a, b)


@torch.inference_mode()
def test_fp8_streaming_loader_and_tied_weights(pair, tmp_path):
    from safetensors.torch import save_file
    from nanovllm.utils.fp8 import load_fp8_model
    ref, _, cfg = pair
    cfg.tie_word_embeddings = True
    ref.tie_weights()
    state, expected = {}, {}
    for name, p in ref.named_parameters():
        if p.ndim == 2:
            quantized = (p*100).to(torch.float8_e4m3fn)
            scale = torch.full(((p.shape[0]+1)//2, (p.shape[1]+1)//2), .01)
            state[name] = quantized
            state[name.removesuffix("weight")+"weight_scale_inv"] = scale
            expected[name] = dequantize_block_fp8(quantized, scale, [2, 2], dtype=torch.float32)
        else:
            state[name] = p.detach().clone()
            expected[name] = p.detach()
    save_file(state, tmp_path/"model.safetensors")
    cfg.quantization_config = {"quant_method": "fp8", "weight_block_size": [2, 2]}
    loaded = load_fp8_model(Qwen3_5ForCausalLM, cfg, tmp_path, "cpu", torch.float32)
    for name, p in loaded.named_parameters():
        torch.testing.assert_close(p, expected[name])
    assert loaded.model.embed_tokens.weight is loaded.lm_head.weight
    assert torch.isfinite(loaded.model.rotary_emb.inv_freq).all()
    assert not loaded.fp8_load_report["fp8_gemm"]


@torch.inference_mode()
def test_compaction_failure_is_atomic(pair):
    _, model, cfg = pair
    cache = HybridCache(cfg, CompressionConfig(budget=4, window=2, pool=1))
    model.forward(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), cache)
    snapshot = cache.snapshot()
    cache.protected_positions = set(range(8))
    with pytest.raises(ValueError):
        cache.compact(force=True)
    for a, b in zip(cache.key_cache, snapshot.key_cache):
        if a is not None:
            assert torch.equal(a, b)
    assert cache.compressions == 0


@torch.inference_mode()
def test_interleaving_is_identical_with_compression(pair):
    _, model, cfg = pair
    def make():
        return HybridCache(cfg, CompressionConfig(budget=4, window=2, interval=1))
    a, b = make(), make()
    ia, ib = make(), make()
    prompts = [torch.tensor([[1, 2, 3, 4, 5, 6]]), torch.tensor([[7, 8, 9, 10, 11, 12]])]
    for state, ids in zip([a, b, ia, ib], prompts*2):
        model.forward(ids, state)
        state.compact(force=True)
    reference = []
    for state in [ia, ib]:
        outputs = []
        for t in [13, 14, 15]:
            outputs.append(model.forward(torch.tensor([[t]]), state))
            state.compact(force=True)
        reference.append(outputs)
    for step, t in enumerate([13, 14, 15]):
        for i, state in reversed(list(enumerate([a, b]))):
            torch.testing.assert_close(model.forward(torch.tensor([[t]]), state), reference[i][step])
            state.compact(force=True)
