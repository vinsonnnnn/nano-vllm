# 文件职责：组织离线生成的完整生命周期；不是 Transformer 网络的计算实现。
# LLM 继承本类；Engine 创建 Sequence，调用 Scheduler 和 ModelRunner，最终解码结果。
# 输入：文字或 CPU token ID 列表；输出：按输入顺序排列的 {text, token_ids} 字典。
# 阅读重点：请求状态在 CPU，模型/KV 在 GPU；step 一次只处理一种阶段的 batch。
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """单机推理控制入口；主进程 rank 0 管理请求，多卡 worker 执行各自模型分片。"""

    def __init__(self, model, **kwargs):
        """初始化配置、TP 进程、模型、tokenizer 和调度器。

        model 为本地模型目录；kwargs 只保留 Config 字段名，其余名称在此被忽略。
        前置条件：CUDA/NCCL/依赖可用，GPU 数及权重结构满足当前 Qwen3 实现要求。
        副作用：启动子进程、分配 GPU 权重/KV/图缓冲，注册进程退出回调。
        """
        # 步骤1：构造配置；Sequence 类级块大小必须与调度器及 Runner 保持一致。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        # 步骤2：spawn 启动 rank 1..P-1，各进程独立初始化 CUDA 和 NCCL。
        # Event/共享内存是 CPU 控制通道；不是传输 GPU hidden states 的通道。
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 步骤3：主进程也执行 rank 0 分片。Runner 会回填 config.num_kvcache_blocks。
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 步骤4：此时 EOS 和 KV 块数已确定，Scheduler 才能使用这些运行参数。
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """广播停止命令、释放 Runner 资源并等待 worker 退出；当前实现不是幂等接口。"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将一条非空输入加入 waiting；字符串先编码，ID 列表直接创建 Sequence。

        不返回生成结果，也不执行 GPU 计算；本方法未实现完整的输入长度校验。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """执行一轮 schedule → run → postprocess，返回已结束结果及本轮统计计数。

        结果为 [(seq_id, completion_ids), ...]；计数正值表示 Prefill 输入量，
        负值表示 Decode 请求数。负号只用于区分阶段，并非真实 token 数为负。
        Scheduler 会更新 Sequence/队列/块元数据，Runner 更新各层 GPU KV。
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 多卡 call 先广播命令再执行本地分片，rank 0 返回每条请求的一个采样 ID。
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """两条请求队列是否都为空；不是判断某个单独 Sequence。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """同步生成全部请求，实际返回 list[dict]（保留原来的 list[str] 类型标注）。

        prompts 为字符串列表或 ID 列表的列表；sampling_params 为一份共享策略
        或逐请求策略列表；use_tqdm 控制进度条。列表长度应由调用者保持一致。
        输出只包含 completion，不包含输入 prompt；不提供流式输出或在线到达模拟。
        """
        # 步骤1：批量入队。重复引用策略对象是安全使用约定，Sequence 随后复制其字段。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 步骤2：迭代调度，每轮可能是 Prefill、部分 Prefill 或 Decode。
        # 进度条只在整条请求结束时更新，不是每生成一个 token 就增加 1。
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 显示的是最近一次对应阶段 step 的速率，而非全程平均吞吐或 TTFT/ITL。
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 步骤3：完成顺序可能不同，用递增 seq_id 恢复本次输入顺序，再在 CPU 解码。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
