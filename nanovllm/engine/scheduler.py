# 文件职责：决定本轮运行哪些请求，并在采样后更新生命周期；不计算 Attention。
# Engine.step 调用 schedule/postprocess；本类调用 BlockManager 分配/复用/回收块。
# 输入输出是 CPU Sequence 和 token IDs；GPU KV 数值由 Runner/Attention 管理。
# 关键概念：Prefill 优先、分块 Prefill、按步重组 batch、队尾抢占和 KV 重计算。
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """管理 waiting/running 两个双端队列；每轮返回一个纯 Prefill 或纯 Decode 批次。

    当前没有显式优先级分数，也没有 Prefill/Decode 混批。Continuous Batching
    指每步可重新选择活跃请求，不等于该离线 generate 已提供在线服务接口。
    """

    def __init__(self, config: Config):
        """读取预算和 EOS，创建 CPU 块管理器；config 的 KV 块数必须已由 Runner 回填。"""
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """两队列均空才结束；有分块 Prefill 留在 waiting 时仍需继续执行。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """新请求追加 waiting 队尾；不为它立刻分配 block。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选本轮请求并预留 KV 槽位，返回 (请求列表, batch 是否为 Prefill)。

        改变队列、状态、num_scheduled_tokens、块表/引用；尚不写入 GPU KV。
        必须存在可推进的请求及正预算，否则末尾 assert 不构成容量恢复方案。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # 步骤1：从 waiting 队头安排 Prefill；remaining 的单位是新输入 token 数。
        # 不跳过一个无法分配的队头去寻找后面的请求，也不为 Prefill 主动抢占 running。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 初次/抢占后重新分配：检查连续前缀命中及整个当前序列的块容量。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 本实现只允许批次的首条请求被切块；后续请求放不完整则结束本轮选择。
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 步骤2：本轮将完成所有未缓存输入时，提前移入 running；GPU 执行在返回后。
            # 未完成的切片仍留在 waiting，但已经持有块表，下一轮接着使用。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # 有任何 Prefill 就直接返回；这是当前 P/D 不混合的明确边界。
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 步骤3：只有没有可执行 Prefill 时才选 Decode；每条请求计算尾 token 的 KV。
        # 队头优先、队尾先被抢占；已经选入 scheduled_seqs 的请求不再在 running 中。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 跨块边界且无空闲块时，释放队尾请求；共享引用可能导致一次释放仍不够。
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # Python while...else：仅在没有执行 break 时进入，表示扩展检查通过。
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 步骤4：保持本轮已选请求的原顺序放回队头，不是轮转到队尾的公平调度。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """在调度边界暂停请求，释放块引用并放 waiting 队头，后续走重计算 Prefill。

        保留 CPU 上 prompt + 已生成 token，不交换 KV 到 CPU，也不取消请求。
        deallocate 重置缓存计数/块表；可能仍可命中尚未被覆盖的空闲前缀块。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """GPU 完成后登记已计算 KV、追加有效采样结果，并检测 EOS/输出上限。

        token_ids 与 seqs 一一对应；中间 Prefill 切片的采样结果会被丢弃。
        请求结束时释放块引用并移出 running，completion 列表保留给 Engine。
        """
        # 步骤1：只对已经实际计算完成的整块登记哈希，然后推进已缓存 token 计数。
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 步骤2：中间切片不是可输出的续写位置；不要把采样 ID 加进请求。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 步骤3：新尾 token 已追加，但它的 KV 通常要到下一轮才计算。
            # 此实现按 == 判断输出上限，调用者应传正整数 max_tokens。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
