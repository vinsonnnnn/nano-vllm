# 文件职责：定义每条请求的生成策略，不负责调度预算或 GPU 内存管理。
# 用户创建 → Sequence 复制字段 → Runner 使用温度，Scheduler 使用终止条件。
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """采样温度、最多新增 token 数和是否忽略 EOS；不是输入 prompt 的长度限制。"""
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """拒绝零/近零温度；当前 Sampler 只有随机采样路径，没有独立 greedy 分支。"""
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
