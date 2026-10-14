# 文件职责：rank 0 根据词表 logits 与温度随机选下一 token；不解码或维护请求队列。
# Runner.run 调用本模块，之后将 GPU 结果转成 CPU list 供 Scheduler 写入 Sequence。
# 输入 GPU logits [B,V]、temperatures [B]；输出 GPU 整型 token IDs [B]。
import torch
from torch import nn


class Sampler(nn.Module):
    """温度缩放 + softmax + 指数随机数竞争的分类采样，不是固定 argmax 贪心。"""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """为 B 条请求各抽取一个 token；温度必须为正，修改中间 logits/probs Tensor。

        logits.float() 在原输入已为 FP32 时可能不复制，div_ 可原地改变传入分数。
        不支持 top-k/top-p，max_tokens/EOS 由 Scheduler 处理，不由此函数判断。
        """
        # 步骤1：[B] 温度扩成 [B,1]，每条请求的一行词表分数使用各自温度。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 步骤2：沿词表 V 归一化；每行概率和约为 1，logits 本身不是概率。
        probs = torch.softmax(logits, dim=-1)
        # 步骤3：为每个候选产生 Exp(1) 随机数，选择 p/e 最大项实现概率采样。
        # argmax 作用于随机化后的值，不是原始 logits；clamp 防止分母接近零。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
