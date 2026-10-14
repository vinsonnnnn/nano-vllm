# 文件职责：提供复制/列并行/行并行和打包投影，用于 Qwen3 Attention 与 MLP。
# loader 通过参数上的 weight_loader 把 CPU 全局权重切片装入 GPU 本卡参数。
# PyTorch 权重布局 [output_size,input_size]：列并行切 dim=0（输出），行并行切 dim=1（输入）。
# 阅读重点：名称来自线性变换的并行划分；不要按 Python 矩阵“行/列”字面混淆切分维。
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """确认维度整除再返回每卡大小；当前不支持用不均匀分片绕过这一约束。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """建立 GPU 参数及本 rank 信息；子类提供具体 forward 和权重加载规则。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """input/output_size 为本卡参数维度，tp_dim 是全局权重加载的切片维。

        在 Runner 设置的默认 CUDA 设备上分配参数；weight_loader 属性供 loader 分派。
        """
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """抽象占位；不能直接调用 LinearBase 执行投影。"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """每 rank 保存完整权重；本类不执行分片或通信，当前 Qwen3 未直接使用它。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """按全局输入/输出宽度创建每卡完整线性层。"""
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """整份 CPU 权重复制到参数设备，无 TP 切片。"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GPU [...,input_size] → [...,output_size]，计算 x @ weight.T + bias。"""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """完整输入、分片输出：全局 W[O,I] → 本卡 W[O/P,I]，forward 不收集输出。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """参数 output_size 是全局输出宽度 O；要求 O 能被 TP 数 P 整除。"""
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """沿输出维 dim=0 取本 rank 连续片段；bias 也按输出维切片。"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整 GPU [...,I] → 本卡 [...,O/P]；后续算子直接消费分片，无 all_gather。"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """将多个列并行投影打包在一个参数中；Qwen3 用它打包 Gate/Up。"""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """output_sizes 保存各全局分支宽度；本卡权重按各分支的局部片段依次排列。"""
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """装入某一分支，loaded_shard_id 如 0=gate/1=up；修改目标打包参数对应视图。

        offset 是前面分支宽度之和/P，size 是当前分支宽度/P；源分支再按 rank 切片。
        """
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """把 Q/K/V 列并行投影打包；GQA 中 Q 与 KV 宽度不同，不能简单平均分三段。"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """以全局头数构造 [Q分片,K分片,V分片]；head_size 是每头维 D。"""
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """把分离的 q/k/v 权重装到对应本卡片段，shard_id 必须为 q、k 或 v。

        每段源权重先按头对应的输出维切 P 份；目标 offset 由本卡 Q/K 宽度计算。
        """
        # 步骤1：定位打包参数中的本卡片段；offset/size 单位是输出特征行数。
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 步骤2：目标取片段视图，源取本 rank 分片，copy_ 写入已分配 GPU 参数。
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """分片输入、完整输出：全局 W[O,I] → 本卡 W[O,I/P]，各卡局部结果求和。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """input_size 是全局 I，调用方必须已经提供本卡 I/P 宽度输入；不自动切 x。"""
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """二维权重按输入 dim=1 切片；一维 bias 每卡复制，forward 仅 rank 0 加一次。"""
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GPU [...,I/P] 投影为局部 [...,O]，all_reduce(SUM) 得完整输出并原地覆盖 y。"""
        # bias 只在 rank 0 加，避免 all_reduce 把同一 bias 重复累加 P 次。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
