# 文件职责：保存一条请求跨多轮的 CPU 状态，不执行模型，也不持有 GPU KV Tensor。
# Engine.add_request 创建；Scheduler 管状态/进度；BlockManager 管块表；Runner 读取输入。
# 关键概念：prompt 与 completion、已缓存与本轮计算 token、逻辑 block、进程快照。
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """调度状态：等待（含分块 Prefill/重计算）、已进入运行队列、已结束。"""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """单条请求的状态容器；token_ids 保存 prompt + 已生成 token 的完整 CPU 列表。

    主进程对象会被调度器更新；多卡 worker 获得的是 __getstate__ 的精简快照，
    并没有所有字段。block_table 的元素是物理 block ID，不是 KV 向量或内存地址。
    """
    # Engine 初始化会用 Config 覆盖类属性；同一进程多引擎可能互相影响。
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """创建等待请求；token_ids 必须非空。复制输入与策略字段，不分配 GPU 缓存。

        seq_id 是进程内递增编号，用于排序结果，不是调度优先级分数。
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        # prompt 长度在生成和抢占后仍保持不变，方便从完整列表切出 completion。
        self.num_prompt_tokens = len(token_ids)
        # 已有有效 KV 的前缀长度；新采样的尾 token 通常尚未计算 KV。
        self.num_cached_tokens = 0
        # Scheduler 本轮安排计算的输入 token 数，不是本轮输出 token 数。
        self.num_scheduled_tokens = 0
        # 用于进程序列化决定传完整列表还是尾 token；不同于整个 batch 的 is_prefill。
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前总 token 数（prompt + completion）；worker 也用这个计数。"""
        return self.num_tokens

    def __getitem__(self, key):
        """按下标/切片读取 token_ids，供 Prefill 打包或按逻辑块提取 token。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """是否已被 Scheduler 标记结束；不在这里检测 EOS 或修改状态。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """返回新增 token 数；num_prompt_tokens 不随抢占或生成变化。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回原始输入的 CPU 列表切片。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回已生成部分的 CPU 列表切片，Engine 用其解码结果。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """按当前总长度向上取整计算逻辑块数；不代表这些块已经分配。"""
        # ceil(n / b) = (n + b - 1) // b。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一逻辑块的 token 数；长度整除时返回 block_size，而不是 0。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个逻辑块的 token 列表，用于前缀哈希；i 必须在有效范围内。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """Scheduler 写入采样结果；只更新列表/尾 token/总长度，不增加已缓存数。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """pickle 发送给 worker 的六元组；Prefill 传完整列表，Decode 只传尾 ID。

        用较小快照传 GPU 输入所需字段，不传请求队列状态、温度和 seq_id。
        温度采样及主队列管理仍由 rank 0 完成。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """从控制消息恢复 worker 快照；Decode 的 token_ids 故意为空，不能完整切片。

        这里不重新调用 __init__，因此 worker 不拥有主进程 Sequence 的全部属性。
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
