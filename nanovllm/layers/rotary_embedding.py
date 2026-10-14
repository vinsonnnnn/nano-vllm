# 文件职责：预计算位置相关 cos/sin，并对 Q/K 施加 RoPE；不旋转 V 或移动缓存块。
# Qwen3Attention 通过 get_rope 创建模块，forward 用 GPU positions [T] 索引缓存。
# 关键概念：按半维配对旋转、头维广播、位置上限、进程内模块复用。
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对 GPU x [T,Nheads,D] 的前/后半维做成对旋转，返回同形状/同 dtype。

    cos/sin [T,1,D/2] 广播到所有头；以 FP32 运算后转回输入 dtype，不原地改 x。
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """保存按位置索引的旋转缓存；当前要求 rotary_dim=head_size，即旋转全部头维。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """head_size/rotary_dim 为 D，max_position_embeddings 为位置缓存长度，base 控制频率。

        在 Runner 的默认 GPU 设备创建 [max_position,1,D] buffer；不是待训练权重。
        persistent=False 表示不写入 state_dict，仍会作为模块 buffer 管理。
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 步骤1：D/2 个频率与所有位置做外积，freqs [max_position,D/2]。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 步骤2：按最后维保存 cos|sin，并增加长度 1 的头轴，以便广播给不同 Q/K 头数。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """用逻辑 positions [T] 旋转 Q [T,Nq,D] 与 K [T,Nkv,D]；位置需在缓存范围内。"""
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """返回位置模块；lru_cache(1) 复用最近一次相同参数实例，不是前缀 KV 缓存。

    缓存键没有设备/dtype，多引擎跨设备复用的适用边界需另行检查。
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
