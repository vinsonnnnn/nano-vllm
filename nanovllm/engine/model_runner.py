# 文件职责：将 CPU 请求整理为 GPU 输入，执行本 rank 的 Qwen3 分片并管理 GPU KV/图。
# Engine 通过 call("run", ...) 调用；Runner 调用 Qwen3、Sampler、Context 和 loader。
# rank 0 返回采样 ID 的 CPU list；worker 只计算模型分片，不自行更新主调度队列。
# 形状约定：B=本轮请求数，T=新输入 token 总数，H=hidden_size，P=TP 进程数。
# 关键概念：变长打包、逻辑 position vs 物理 slot、NCCL 数据通信 vs 共享内存控制。
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """每个 TP rank 一个执行器；GPU 权重/KV 独立，CPU Context 也按进程独立。

    本实现固定构造 Qwen3ForCausalLM，不是按 architectures 自动分发任意模型。
    单卡也初始化 NCCL；多卡 rank 0 负责控制广播和最终采样。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """在对应 GPU 上加载模型、预热、分配 KV 并可选捕获 Decode 图。

        rank 0 的 event 是各 worker 的 Event 列表，其余 rank 是一个 Event。
        需要各 rank 同序进入 NCCL 集合操作；非零 rank 初始化末尾进入命令循环。
        修改 config.num_kvcache_blocks，创建权重/KV/图及多卡共享内存资源。
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 步骤1：用 NCCL 建立 GPU 通信组，固定 localhost 端口用于单机 rendezvous。
        # rank 对应本机 CUDA 设备编号；同时启动多个引擎可能发生端口冲突。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 步骤2：临时把默认设备/dtype 设为模型 GPU 类型，让参数及缓存在正确设备创建。
        # torch.empty 的参数只是分配空间，必须加载权重后才执行真实推理。
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 步骤3：先测无 KV 的 Prefill 峰值显存，再估算剩余预算并分配持久 KV 池。
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 步骤4：1 MiB 命名共享内存传 pickle 命令。barrier 保证创建后 worker 才连接。
        # GPU 张量的 all_reduce/gather 走 NCCL，不走这块 CPU 共享内存。
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """关闭控制通道/图引用、同步 GPU 并销毁进程组；各 rank 必须协作进入此流程。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """worker 等待并执行与 rank 0 相同的方法；收到 exit 后退出控制循环。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """worker 读控制消息，返回 (方法名, 参数列表)；等待 Event 后读取并清除通知。"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        # 前四字节是 payload 字节数，之后才是 pickle 内容；这不是 token 数。
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """rank 0 将方法名和实参序列化到共享内存，再唤醒全部 worker。

        *args 在定义处收集位置参数；Sequence 的 pickle 钩子缩小传输内容。
        当前固定缓冲区没有分包协议，消息必须装得下；不用于不可信外部输入。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """按字符串找到本地方法并执行；多卡 rank 0 先通知 worker 同步执行。

        例如 call("run", seqs, True) 对应 self.run(seqs, True)。调用处 *args
        展开参数元组；这是同机多进程控制，不是 HTTP RPC。未知方法未专门校验。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """用合成 CPU Sequence 跑 Prefill，触发算子/编译并记录峰值；此时尚无 KV 池。

        模型在 GPU 执行，空块表不生成写缓存槽位，Attention 的空缓存跳过 KV 写入。
        num_seqs/seq_len 同时受 token 预算、长度和请求数限制，不是实际用户请求。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """按显存预算估算块数，创建 GPU KV 池并将各层的 k/v_cache 绑定为池的视图。

        池形状 [2,L,Nb,S,Nkv,D]：K/V、层、物理块、块内 token、每卡 KV 头、头维。
        只分配空间，不计算 KV；预算估算不保证所有配置下都不会 OOM。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 步骤1：算一个物理块跨所有层的字节数；2 是 K/V，itemsize 是每元素字节数。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # used 包含当前设备已用显存，peak-current 预留预热测得的临时分配峰值增量。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 步骤2：绑定各层 [Nb,S,Nkv,D] 视图，避免每层再次分配/复制 KV 池。
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把 B 条 CPU 块表补齐成 GPU int32 Tensor [B,max_blocks]；-1 为补齐哨兵。

        最大块表长度来自本轮请求；实际有效范围由各请求的上下文长度约束。
        pin_memory + non_blocking 便于主机到设备复制，不代表整个推理无同步。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """打包新 Prefill 切片，返回 GPU int64 input_ids/positions [T] 并设置 Context。

        输入 Sequence 已有 num_scheduled_tokens 和块表（预热例外）。Q 只含新输入，
        K 长度包括已缓存前缀；cu_seqlens_q/k 为 GPU int32 [B+1] 累积长度。
        slot_mapping 为 GPU int32 [T]；存在历史 KV 时还提供 [B,max_blocks] 块表。
        """
        # 步骤1：先用 CPU 列表按请求顺序拼接输入，各条请求位置从其真实缓存进度开始。
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # [start,end) 是本轮计算区间；seqlen_q=end-start，seqlen_k=end。
            # 例如已缓存 512、新算 88：Q 长 88，K 长 600，不是都等于 88。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            # 累积边界分隔拼接后的请求，如输入长度 3/2 得到 [0,3,5]。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 步骤2：按块展开物理槽位。positions 是句子内位置，slot 是池中位置。
            # floor(start/S)、ceil(end/S) 找出切片涉及的逻辑块半开区间。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # slot = 物理 block ID * S + 块内偏移，单位是 token 槽，不是字节。
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # 步骤3：有已缓存历史（前缀命中或之前的 Prefill 切片）时从 paged KV 读取 K/V。
        # 因此原来的 prefix cache 注释不只覆盖哈希命中，还覆盖后续切片。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # 步骤4：CPU 列表 → pinned CPU Tensor → GPU Tensor；设置进程内本轮 Context。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """每请求取一个尚未计算 KV 的尾 token，返回 GPU int64 [B] 输入/位置。

        slot_mapping/context_lens 是 GPU int32 [B]，block_tables 是 [B,max_blocks]。
        上一轮采样的 token 已追加在 CPU 列表中；本轮写它的 KV，再预测下一个。
        """
        # 步骤1：尾 token 的逻辑 position 为 len-1；历史读取长度 len 包含该尾 token。
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 最后物理块起始槽 + 最后块 token 数 - 1，数量要转换成零起始偏移。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """rank 0 把逐请求温度转成 GPU float32 [B]，Sampler 用它按行缩放 logits。"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行模型主干及 LM Head；rank 0 返回 GPU logits [B,V]，worker 多卡返回 None。

        Prefill、enforce_eager 或 Decode B>512 走普通调用，其余 Decode 重放图。
        普通调用仍可能触发各层 @torch.compile；不等于全部算子都未经编译。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 步骤1：选容量不小于实际 B 的最小图；未覆盖的自定义上限会令 next 失败。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 步骤2：只更新固定 GPU 地址里的值，不重新分配图捕获使用的输入缓冲区。
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # 图容量可能大于 B：无效行的槽位设 -1，kernel 跳过写；读取长度设 0。
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 步骤3：重放的是本轮新数据的主干计算，不复用旧答案；LM Head 在图外执行。
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """准备输入 → 执行主干/输出头 → rank 0 采样 → 清理临时 Context。

        rank 0 将 GPU [B] token IDs 转成 CPU list[int]，会读取 GPU 结果；worker
        实际返回 None（保留原类型标注）。不在这里解码文字或更新请求状态。
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 仅换掉本轮元数据，不释放持久 KV；当前不是 try/finally，异常时不保证重置。
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """初始化时为若干 Decode batch 容量捕获 GPU 模型主干，并保存静态输入/输出。

        输入/位置/槽位/长度 [max_bs]、块表 [max_bs,max_num_blocks]、输出 [max_bs,H]。
        不捕获调度器、CPU 控制消息、LM Head、Sampler 或 tokenizer；仅适用纯 Decode。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 步骤1：默认设备此时仍为 CUDA；以下缓冲区在初始化后长期保留固定地址。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 步骤2：少量容量档位避免为每一个 B 捕获；实际 batch 通过无效行补齐。
        # 当前档位生成未处理所有小/非 16 倍数的 max_bs 边界，见阅读指南潜在问题。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 步骤3：每个容量先预热后 capture，共用图内存池；各 rank 捕获相同通信顺序。
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
