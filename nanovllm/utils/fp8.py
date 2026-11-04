"""Explicit block-FP8 checkpoint -> BF16 loading, not FP8 GEMM.

Supported contract: quant_method=fp8, weight_block_size=[rows, cols], E4M3FN,
and weight_scale_inv storing the dequantization multiplier (HF fine-grained
FP8 convention). Other quantization formats fail instead of guessing scales.
"""
from pathlib import Path
import torch
from safetensors import safe_open


def dequantize_block_fp8(weight, scale, block_size, *, shard_dim=None, rank=0, world_size=1,
                         dtype=torch.bfloat16):
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("block FP8 requires matrix weights and a 2D scale grid")
    rows, cols = block_size
    if min(rows, cols, world_size) < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid FP8 block size or TP rank")
    expected = ((weight.shape[0]+rows-1)//rows, (weight.shape[1]+cols-1)//cols)
    if tuple(scale.shape) != expected or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid FP8 scale grid")
    if shard_dim is not None:
        if shard_dim not in (0, 1) or weight.shape[shard_dim] % world_size:
            raise ValueError("TP weight dimension must divide world_size")
        width = weight.shape[shard_dim] // world_size
        block = (rows, cols)[shard_dim]
        if width % block:
            raise ValueError("TP shard boundaries must align with FP8 scale blocks")
        weight = weight.narrow(shard_dim, rank * width, width)
        scale = scale.narrow(shard_dim, rank * width // block, width // block)
    expanded = scale.float().repeat_interleave(rows, 0).repeat_interleave(cols, 1)
    return (weight.float() * expanded[:weight.shape[0], :weight.shape[1]]).to(dtype)


def load_fp8_model(model_cls, config, path, device, dtype):
    quant = config.quantization_config
    quant = quant if isinstance(quant, dict) else quant.to_dict()
    if quant.get("quant_method") != "fp8" or not quant.get("weight_block_size"):
        raise NotImplementedError("only HF block-scaled FP8 checkpoints are supported")
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("no local safetensors checkpoint shards")
    locations = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise ValueError(f"duplicate checkpoint key: {key}")
                locations[key] = file
    with torch.device("meta"):
        model = model_cls(config)
    model.to_empty(device=device)
    model.to(dtype=dtype)
    model.tie_weights()
    loaded, ignored = set(), []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in locations:
                raise ValueError(f"missing checkpoint weight: {name}")
            with safe_open(locations[name], framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(name)
            loaded.add(name)
            if weight.dtype == torch.float8_e4m3fn:
                scale_name = name.removesuffix("weight") + "weight_scale_inv"
                if scale_name not in locations:
                    raise ValueError(f"missing FP8 multiplier: {scale_name}")
                with safe_open(locations[scale_name], framework="pt", device="cpu") as handle:
                    scale = handle.get_tensor(scale_name)
                weight = dequantize_block_fp8(weight, scale, quant["weight_block_size"], dtype=dtype)
                loaded.add(scale_name)
            elif weight.dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2):
                raise NotImplementedError(f"unsupported FP8 dtype: {weight.dtype}")
            if tuple(weight.shape) != tuple(parameter.shape):
                raise ValueError(f"shape mismatch for {name}")
            parameter.copy_(weight)
        # Check tied aliases too. MTP weights are not silently called supported.
        aliases = dict(model.named_parameters(remove_duplicate=False))
        for name in locations.keys() - loaded:
            if name in aliases and id(aliases[name]) in {id(p) for p in model.parameters()}:
                continue
            if name.startswith("mtp.") or name.startswith("model.mtp."):
                ignored.append(name)
            else:
                raise ValueError(f"unexpected checkpoint weight: {name}")
    # to_empty also empties nonpersistent RoPE buffers: rebuild rather than use garbage.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, Qwen3_5VisionRotaryEmbedding
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    text.rotary_emb = Qwen3_5TextRotaryEmbedding(text.config, device=device)
    if hasattr(model.model, "visual"):
        vision = model.model.visual
        head_dim = vision.config.hidden_size // vision.config.num_heads
        vision.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2).to(device)
    model.fp8_load_report = {"loaded": len(loaded), "unused_mtp_keys": ignored,
                             "execution_dtype": str(dtype), "fp8_gemm": False}
    return model
