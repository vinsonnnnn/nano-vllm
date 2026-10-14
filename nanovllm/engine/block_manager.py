# 文件职责：管理物理 KV block 的 CPU 元数据、引用与前缀哈希，不存储/计算 K/V Tensor。
# Scheduler 调用分配/扩展/回收/哈希；Runner 按同样的 block ID 布局 GPU KV 池。
# Sequence.block_table 把逻辑块映射到物理块，Runner 再生成逐 token 的 slot_mapping。
# 关键概念：free 不等于 KV 内容清零；引用计数为零的块可在被覆盖前继续充当前缀缓存。
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """一个物理块的管理记录：ID、活跃引用数、前缀哈希和用于核对命中的 token 列表。"""

    def __init__(self, block_id):
        """初始化空闲记录；hash=-1 表示没有已登记的有效整块前缀。"""
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """计算完成后记录整块前缀哈希/token；不更改引用数或 GPU KV 内容。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """分配给新内容时设一个引用并清除旧哈希元数据；实际 KV 将由 Attention 覆写。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """固定物理块池的管理器；used/free 按是否存在活跃引用划分，不是按是否有历史 KV。

    hash_to_block_id 同时可索引 used 和 free 的完整前缀块；它不是 GPU 数据副本。
    依赖单控制线程按 schedule → GPU 完成 → postprocess 顺序调用，不提供并发锁。
    """

    def __init__(self, num_blocks: int, block_size: int):
        """创建 num_blocks 条 CPU 记录；block_size 的单位是每块可放的 token 数。"""
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """返回 64 位整型哈希：父前缀哈希字节 + 当前块 token 字节。

        prefix=-1 代表第一块。把前缀链纳入哈希，是因为同样的当前 token
        在不同历史上下文中通常产生不同 KV。这里只使用哈希，不计算神经网络。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """取 free 队头给新内容，失效旧哈希映射，设 ref_count=1 并返回物理 ID。

        前置条件是至少有一个空闲块；不分配新的 CUDA Tensor。
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 同哈希映射可能已指向另一块，因此只有仍指向本块时才删除该映射。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """最后一个引用释放后归还 free 队尾；保留哈希/token 和未被覆盖的 GPU KV。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """只检查、不修改状态：可容纳则返回连续命中块数，容量不足返回 -1。

        仅检查最后一个逻辑块之前的连续整块；最后一块即使满也不在此复用。
        整个当前序列的 block 容量都要满足，不是只为本轮 Prefill 切片预留。
        """
        # 步骤1：沿父哈希链查找连续前缀；第一次缺失/当前块 token 不符即停止。
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 已 used 的命中块共享引用，不再消耗 free；free 命中仍占用一个空闲名额。
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 步骤2：检查未命中块 + 需要重新激活的 free 命中块是否有足够容量。
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为无块表请求建立逻辑→物理映射，并回填命中前缀 token 数。

        num_cached_blocks 来自紧接着执行的 can_allocate；中间不能有并发池变更。
        命中 used 增引用，命中 free 重新激活，其余块为新内容分配；不写 GPU KV。
        """
        # 步骤1：命中块沿相同哈希链恢复；共享的是同一物理 KV，不是复制一份。
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 步骤2：为所有剩余逻辑块分配物理 ID，包括还没轮到本轮切片计算的块。
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放该请求全部块引用，清空块表和缓存计数，但保留 CPU token 列表。

        只有 ref_count 降到零的物理块才可重用；不会 cudaFree 整个 GPU KV 池。
        被抢占请求之后可重算，也可能复用还未被覆盖的前缀缓存。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Decode 前检查尾 token 是否有槽位；跨到新块第一个 token 才需一个 free 块。"""
        # bool 在比较中为 0/1：len % block_size == 1 时需 1 个，否则需 0 个。
        # 尾 token 已由上一轮采样追加，尚未计算 KV；例如块大小 256、总长 257。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """在 can_append 通过后，必要时为 Decode 尾 token 增加一个物理块。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """GPU 本轮执行成功后登记新完成的整块；部分块不登记，不提前宣称缓存有效。

        起止范围由执行前缓存计数和本轮计算数量决定；更新 Block 及哈希索引。
        """
        # 步骤1：floor 前后缓存 token 数得到新完成块的半开区间 [start, end)。
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 步骤2：接续上一完整块的父哈希；从第一块开始时使用 -1 哨兵。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
