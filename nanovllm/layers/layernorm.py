# 文件职责：RMSNorm 归一化及残差加法，供 Decoder、最终 Norm 和 Q/K Norm 使用。
# 沿最后一维处理 GPU 特征；H 维输入 [T,H] 或单头 D 维输入 [T,Nheads,D]。
# 关键概念：均方根不减均值、FP32 中间计算、可训练逐维缩放、原地运算与别名。
import torch
from torch import nn


class RMSNorm(nn.Module):
    """按最后一维的均方根调整尺度，再乘逐维 weight；不把数据变成概率分布。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """hidden_size 为被归一化的最后一维；eps 防止除零，weight 后续由 loader 加载。"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """不带残差的归一化：GPU x [...,H] → 同形状、原 dtype Tensor。

        方差变量 var 实际是 mean(x²)，不是减均值后的统计方差。
        float() 在输入已为 FP32 时可能保留同一存储，后续原地操作需注意别名。
        """
        # 步骤1：提升中间计算精度；每行仅沿最后一维求均方，保留广播维。
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 步骤2：乘 1/sqrt(mean(x²)+eps)，还原 dtype 后乘可学习的逐维缩放。
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """合并分支 x 与 residual 后归一化，返回 (归一化结果, 更新后的残差)。

        两输入形状相同；本函数含原地操作。FP32 下 x.to(orig_dtype) 可不复制，
        residual 与中间 x 可能共享存储，不能无条件假定残差是一份独立快照。
        """
        # 步骤1：在 FP32 中执行残差加法；低精度时 residual 转回原 dtype 通常形成新存储。
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """无 residual 返回单 Tensor，有 residual 返回二元组；调用方需匹配该返回约定。"""
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
