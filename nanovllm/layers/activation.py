# 文件职责：为 Qwen3MLP 实现门控激活 SiLU(gate) * up；不包含前后线性投影。
# 输入 GPU [T,2I/P]，输出 [T,I/P]；@torch.compile 优化这个局部计算。
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """把打包 Gate/Up 输入分成两段，用 SiLU 激活 Gate 后逐元素乘 Up。"""

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """最后一维应含两段等宽特征；chunk 不混合请求，只分隔特征维。"""
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
