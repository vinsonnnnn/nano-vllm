# 文件职责：把本轮 K/V 写入物理槽，再调用外部 FlashAttention 完成注意力计算。
# Qwen3Attention 调用 Attention.forward；位置旋转已在调用前完成，这里不做 RoPE。
# 输入 Q [T,Nq,Dh]、K/V [T,Nkv,Dh]；各层 KV 视图 [Nb,S,Nkv,Dh] 位于 GPU。
# 关键概念：slot_mapping 负责写入，block_tables/长度负责读取，Q 头数可大于 KV 头数。
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton 写缓存 kernel：每个 program 负责一个输入 token 的全部本卡 KV 头。

    指针指向 GPU 数据；stride 和偏移的单位是元素而非字节；D=Nkv*Dh。
    slot=-1 为 CUDA Graph 无效填充行，直接跳过，避免覆盖有效 KV。
    """
    # 步骤1：网格 (N,) 的 program ID 就是本轮输入 token 下标，不是 token ID 数值。
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 步骤2：K/V 可能来自打包 QKV 的视图，token 间 stride 不一定等于 D。
    # 头内数据需连续；用各自 stride 找到输入 token，再读取该 token 的 D 个元素。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 步骤3：物理槽编号已将 block 和块内偏移展平；一个槽含 D 个 K 元素及 D 个 V 元素。
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """校验 GPU K/V 布局并启动 N 个 program；修改缓存，不返回数据。

    key/value [N,Nkv,Dh]，slot_mapping [N]，缓存 [Nb,S,Nkv,Dh]。
    N 是本轮输入 token 总数；这里的 num_heads 指 KV 头，而非 Q 头。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """本层注意力与 KV 访问；缓存 Tensor 由 Runner 绑定，块分配不在本类执行。"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """记录本卡头数、头维及 QK 缩放；先用空缓存支持分配 KV 池之前的预热。"""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """先写本轮 KV，再按 Context 选择 Prefill/Decode 接口，返回 GPU 头输出。

        Prefill 输出通常 [T,Nq,Dh]；Decode q 扩成 [B,1,Nq,Dh]，对应单 query。
        有历史缓存时从 paged KV 读取；无历史 Prefill 可直接使用本轮 K/V。
        """
        # 步骤1：缓存存在才写入；warmup 的空 Tensor 走不写缓存路径。
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # 步骤2：Prefill 用变长接口。causal=True 屏蔽未来位置，不让请求互相串扰。
        # q/k 累积长度分别描述新 query 与包括历史的 key 长度，不总是相等。
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        # 步骤3：Decode 的 context_lens 包含本轮尾 token，其 KV 在步骤1已经写入。
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
