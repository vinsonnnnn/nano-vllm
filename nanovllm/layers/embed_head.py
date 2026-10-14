# 文件职责：提供输入词嵌入和输出词表打分，均沿词表维度做张量并行切分。
# Qwen3Model 调 Embedding，Qwen3ForCausalLM.compute_logits 调 LM Head；后者读取 Context。
# 关键形状：全局权重 [V,H]，本卡 [V/P,H]；输入 ID [T]，特征 [T,H]，logits [B,V]。
# 词嵌入是查表、LM Head 是矩阵乘法；两者即使共享权重，也不是同一种运算。
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """每卡持有连续一段词表；输入 ID 对应的向量最终 all_reduce 合并为完整特征。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """num_embeddings=全局 V，embedding_dim=H；要求 V 整除 TP 数 P，进程组已初始化。"""
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从 CPU 全局权重的词表维切出本 rank 区间，复制到 GPU 本卡 [V/P,H] 参数。"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """GPU token IDs [T] 查表得到 [T,H]；多卡仅本卡负责的 ID 贡献非零向量。"""
        # 步骤1：全局 ID 转为本卡局部 ID；范围外暂用 0，避免查表越界。
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        # 步骤2：按行读取向量，不是对整数 ID 做线性乘法。
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 步骤3：清零非本卡词表的结果，求和后每个 token 只保留负责卡的向量。
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """每卡计算本词表分片的分数，rank 0 gather 后拼成完整 logits；不选择 token。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """复用词表分片参数布局/加载器；本项目禁止输出头 bias。"""
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """输入最终特征；Prefill [T,H] 先选各请求末行，Decode 已是 [B,H]。

        本卡 logits [B,V/P]；rank 0 返回 [B,V]，多卡其余 rank 返回 None。
        中间 Prefill 切片也会打分，但它的采样 ID 随后被 Scheduler 丢弃。
        """
        context = get_context()
        if context.is_prefill:
            # 步骤1：[0,3,5] → 去掉首边界 [3,5] → 减 1 得末行下标 [2,4]。
            # 选的是本轮各 query 切片末行，未必是整个 prompt 末行。
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # 步骤2：x @ weight.T，给每个候选 token 打未归一化分数；不是概率。
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 步骤3：按 rank 对应词表区间收集，并沿最后一维拼接；不是把分数求和。
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
