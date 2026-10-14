# 文件职责：将引擎资源预算与模型配置汇集成 Config，供 Engine/Runner/Scheduler 使用。
# LLMEngine 创建 Config；AutoConfig 读取本地模型配置，但不加载权重数值。
# 关键概念：token 预算、请求数上限、TP 进程数、KV 块大小；计数单位不要混淆。
import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """引擎配置及初始化过程中回填的运行参数。

    model 是本地目录；hf_config 包含 H/层数/头数/dtype 等模型结构参数。
    eos 由 tokenizer 回填，num_kvcache_blocks 由 Runner 按显存测量结果回填。
    本类不提供完整的请求长度、显存预算或模型架构兼容性验证。
    """
    model: str
    # Prefill 一轮新计算 token 的总预算；当前 Decode 分支没有使用此预算。
    max_num_batched_tokens: int = 16384
    # 每轮选出的请求数量上限，并非整个引擎能接收的请求总量。
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # 跳过 CUDA Graph 捕获/重放；不会移除计算层上的 @torch.compile。
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    # 一个物理 block 为所有层保留 block_size 个 token 的 K/V；池大小初始化后确定。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """dataclass 初始化后校验路径/块大小/TP，读取模型配置并收紧 max_model_len。"""
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 这里只调整配置值；不等于 add_request 已执行输入/总长度边界检查。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
