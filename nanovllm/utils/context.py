# 文件职责：保存当前 batch 的执行说明，避免沿每层 forward 反复传递所有辅助参数。
# Runner 设置；Attention/LM Head 读取；run 完成后重置。不是聊天记录或 with 管理器。
# 关键概念：进程内全局变量，不是跨进程共享，也没有并发 forward 的上下文隔离。
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """本轮临时元数据容器；不存储 token 内容、模型权重或实际 KV 数值。

    B=请求数，T=本轮 query token 总数；Tensor 通常由 Runner 放到 GPU。
    Prefill 用 cu_seqlens，Decode 用 context_lens；未使用字段允许为 None。
    """
    is_prefill: bool = False
    # GPU int32 [B+1]：拼接 query 的累积边界，如 [0,3,5] 分隔两条请求。
    cu_seqlens_q: torch.Tensor | None = None
    # GPU int32 [B+1]：各请求历史 + 本轮新输入的 key 长度的累积值。
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # GPU int32 [T]（Decode T=B）：本轮每个 token 的物理 KV 写入槽；-1 跳过。
    slot_mapping: torch.Tensor | None = None
    # GPU int32 [B]：Decode 当前上下文长度，包含本轮尾 token。
    context_lens: torch.Tensor | None = None
    # GPU int32 [B,max_blocks]：各请求逻辑块 → 物理 ID，按行补齐。
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    """返回本进程当前对象，不复制；计算层按约定只读取，避免互相修改本轮说明。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """用参数构造新 Context 并替换全局引用；字段含义见 Context，不复制 GPU Tensor。

    Runner 在计算前调用；CUDA Graph 捕获期间同样调用，以提供静态 buffer 视图。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """换成默认空说明；不清空 GPU KV，也不释放请求的 block 或 token 列表。"""
    global _CONTEXT
    _CONTEXT = Context()
