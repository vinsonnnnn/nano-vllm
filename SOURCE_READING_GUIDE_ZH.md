# Nano-vLLM 中文源码阅读指南

本指南对应本仓库当前代码，教学注释基线为提交 `59d099d`，整理日期为 2026-10-14。此次只添加中文注释、docstring 和阅读文档，不重构或优化执行逻辑。

已有的 [从 0 到 1 教程](NANO_VLLM_ZERO_TO_ONE_ZH.md) 解释基础概念；本指南帮助你沿实际文件、函数和状态变化阅读源码。链接不依赖行号：打开文件后搜索类名或函数名即可。

## 1. 先建立模块关系

把工作分成两类：**CPU 管请求和资源，GPU 算模型和缓存数据**。模型 GPU 执行前后的整理、采样结果读取和文字解码，仍然需要 CPU 参与。

```text
用户：example.py / bench.py
    │ LLM.generate(...)
    ▼
LLM → LLMEngine（入口、结果、生命周期）
    ├─ Config / SamplingParams（资源预算 / 每请求策略）
    ├─ Tokenizer → Sequence（输入编码、单条请求状态）
    ├─ Scheduler（本轮选哪些请求、计算多少 token）
    │     └─ BlockManager → Block（物理块引用、前缀哈希、块表）
    └─ ModelRunner（每 rank 的 GPU 执行器）
          ├─ Context（本轮边界、长度、缓存映射）
          ├─ load_model（加载并切分权重）
          ├─ Qwen3ForCausalLM
          │     ├─ Qwen3Model
          │     │     └─ Embedding → Decoder Layers → RMSNorm
          │     │           └─ Attention / MLP / RoPE / Linear
          │     └─ ParallelLMHead → logits
          └─ rank 0 Sampler → 新 token ID → Scheduler 写回
```

先牢牢记住四个区别：

- `Sequence` 是一条请求跨多轮的状态；`Context` 是本轮整个 batch 的执行说明。
- `Block` 是 CPU 元数据；真正的 KV 数值存放在 `ModelRunner.kv_cache` 的 GPU Tensor 中。
- `block_table` 按逻辑块寻找物理块；`slot_mapping` 按本轮输入 token 指定 KV 写入槽位。
- LM Head 负责打分，Sampler 负责选择，Scheduler 负责追加和终止，Tokenizer 负责解码。

## 2. 推荐阅读顺序

建议每读完一项，回答右侧的问题，再进入下一项。不必第一次就理解所有 GPU kernel。

| 顺序 | 文件与建议搜索的符号 | 阅读重点 | 自我检验问题 |
|---|---|---|---|
| 1 | [example.py](example.py)：`main`；[bench.py](bench.py)：`main` | 用户怎样构造 prompt、策略并调用 generate | 聊天模板返回的是字符串还是 ID？benchmark 的计时包含什么？ |
| 2 | [__init__.py](nanovllm/__init__.py)、[llm.py](nanovllm/llm.py)：`LLM` | 公开入口只是导出/继承，实际实现在哪里 | 为什么 LLM 中只有 pass 也能调用 generate？ |
| 3 | [config.py](nanovllm/config.py)：`Config`；[sampling_params.py](nanovllm/sampling_params.py)：`SamplingParams` | 引擎资源预算与单请求生成策略 | max_num_batched_tokens 与 max_tokens 有什么区别？ |
| 4 | [sequence.py](nanovllm/engine/sequence.py)：`Sequence`、`append_token`、序列化钩子 | CPU token 列表、进度、块表、worker 快照 | 新采样 token 为什么不立即增加 num_cached_tokens？ |
| 5 | [llm_engine.py](nanovllm/engine/llm_engine.py)：`generate` → `add_request` → `step` | 从输入到返回结果的组织流程 | 请求完成顺序与结果返回顺序为什么可以不同？ |
| 6 | [scheduler.py](nanovllm/engine/scheduler.py)：`schedule`、`preempt`、`postprocess` | 两条队列、切片预算、抢占、结束判定 | 为什么中间 Prefill 切片的采样不能追加进请求？ |
| 7 | [block_manager.py](nanovllm/engine/block_manager.py)：`can_allocate`、`allocate`、`hash_blocks`、`deallocate` | 物理块容量、共享引用、连续前缀命中 | 空闲块为什么仍可能命中缓存？共享块释放一次为什么未必变空闲？ |
| 8 | [model_runner.py](nanovllm/engine/model_runner.py)：`run`、`prepare_prefill`、`prepare_decode` | CPU 数据打包到 GPU、本轮元数据 | positions 与 slot_mapping 为什么可能完全不同？ |
| 9 | [context.py](nanovllm/utils/context.py)：`Context`、`set_context`、`get_context` | 长度/边界/映射的生产与消费 | reset_context 为什么不能清空历史 KV？ |
| 10 | [qwen3.py](nanovllm/models/qwen3.py)：`Qwen3Model.forward`、`Qwen3Attention.forward`、`Qwen3MLP.forward` | 模型主干、头维形状、残差流 | Q 头数能否与 KV 头数不同？完整 Sequence 是否传进网络？ |
| 11 | [attention.py](nanovllm/layers/attention.py)：`Attention.forward`、`store_kvcache_kernel` | 本轮写 KV、历史读取、FlashAttention | kernel 的 idx 是 token ID，还是本轮 token 的下标？ |
| 12 | [rotary_embedding.py](nanovllm/layers/rotary_embedding.py)：`RotaryEmbedding`、`apply_rotary_emb` | 位置索引、Q/K 旋转、头轴广播 | 为什么不在这个模块旋转 V？ |
| 13 | [layernorm.py](nanovllm/layers/layernorm.py)：`RMSNorm`；[activation.py](nanovllm/layers/activation.py)：`SiluAndMul` | 最后维归一化、残差合并、门控激活 | var 真的是减均值后的方差吗？Gate/Up 沿哪个维度切开？ |
| 14 | [embed_head.py](nanovllm/layers/embed_head.py)：`VocabParallelEmbedding`、`ParallelLMHead` | token 查表、末 query 行选择、词表打分 | 相同权重为什么能分别用于查表和线性变换？ |
| 15 | [sampler.py](nanovllm/layers/sampler.py)：`Sampler.forward` | 温度、softmax、随机采样 | 最后使用 argmax 为什么仍然不是 greedy？ |
| 16 | [loader.py](nanovllm/utils/loader.py)：`load_model`；[linear.py](nanovllm/layers/linear.py)：各 `weight_loader` | CPU 权重 → 打包参数 → 本 rank GPU 分片 | q_proj 的独立权重怎样进入 qkv_proj？shard_id 与 rank 是否相同？ |
| 17 | [linear.py](nanovllm/layers/linear.py)：列并行、行并行 `forward`；[embed_head.py](nanovllm/layers/embed_head.py) | 哪些运算需要 all_reduce/gather | 为什么 RowParallelLinear 只在 rank 0 加 bias？ |
| 18 | [model_runner.py](nanovllm/engine/model_runner.py)：`write_shm`、`read_shm`、`call`、`loop`、`exit` | 多进程控制与 NCCL 同序执行 | Event 通知与 NCCL 张量通信各自传什么？ |
| 19 | [model_runner.py](nanovllm/engine/model_runner.py)：`warmup_model`、`allocate_kv_cache`、`capture_cudagraph`、`run_model` | 峰值显存预算、静态 buffer、Decode 图重放 | CUDA Graph 捕获是否包含 LM Head、Sampler 和 CPU 调度？ |

## 3. 端到端调用链

### 3.1 初始化：先准备资源，不是已经开始用户生成

```text
LLM(model_path, **kwargs)
  → LLMEngine.__init__
      → Config.__post_init__ → AutoConfig 读取本地模型结构
      → spawn rank 1..P-1 的 ModelRunner
      → rank 0 ModelRunner.__init__
          → 初始化 NCCL / 设置当前 CUDA 设备
          → Qwen3ForCausalLM → load_model
          → warmup_model → allocate_kv_cache
          → 可选 capture_cudagraph
      → AutoTokenizer → 回填 EOS
      → Scheduler → CPU BlockManager
```

`num_kvcache_blocks` 是 Runner 根据实际显存测量结果回填的。因此先创建 Runner，再创建 Scheduler，后者才知道物理块总量。

### 3.2 请求执行：一轮生成一个候选，不是一轮完成全文

```text
LLMEngine.generate
  → add_request
      → 字符串 tokenizer.encode（已提供 IDs 则跳过）
      → Sequence → Scheduler.add → waiting
  → 循环 step，直到 waiting/running 都为空
      → Scheduler.schedule → (seqs, is_prefill)
      → ModelRunner.call("run", seqs, is_prefill)
          → 多卡先 write_shm 通知 worker
          → run → prepare_prefill / prepare_decode → set_context
          → run_model
              → 普通调用 model 或 Decode graph.replay
              → compute_logits → ParallelLMHead
          → rank 0 Sampler → GPU token IDs → .tolist() 到 CPU
          → reset_context
      → Scheduler.postprocess
          → hash_blocks / 推进缓存计数
          → 丢弃中间切片候选，或追加新 token
          → EOS/上限检查 → 回收结束请求
  → 按 seq_id 排序 → tokenizer.decode → 返回字典列表
```

注意：`Qwen3ForCausalLM.forward` 返回 hidden states，`compute_logits` 才执行输出头。CUDA Graph 捕获前者，不包含后者及采样。

## 4. 手工追踪一条请求

### 4.1 常规 Prefill → Decode → 结束

假设输入编码为 `[10,20,30]`，`max_tokens=2`，模型依次采样得到 `40`、`50`。这些 ID 和结果是教学假设。

| 时刻 | token_ids | num_tokens | num_cached_tokens | num_scheduled_tokens | 队列/状态 |
|---|---|---:|---:|---:|---|
| 创建 | `[10,20,30]` | 3 | 0 | 0 | waiting / WAITING |
| schedule Prefill 后 | `[10,20,30]` | 3 | 0 | 3 | 本轮将完成输入，已移 running |
| GPU Prefill 完成并 postprocess | `[10,20,30,40]` | 4 | 3 | 0 | running / RUNNING |
| schedule Decode 后 | `[10,20,30,40]` | 4 | 3 | 1 | 本轮计算尾 token 40 |
| GPU Decode 完成、生成 50 并结束 | `[10,20,30,40,50]` | 5 | 0 | 0 | FINISHED，移出 running |

逐步理解：

1. `add_request` 创建对象，暂不执行模型或分配实际 KV 数据。
2. `schedule` 先建立块表并安排输入量。状态可能在 GPU 执行之前就改为 RUNNING；它是调度状态，不是 CUDA 完成信号。
3. Prefill 输入 `10、20、30`，计算这三个 token 的各层 KV。LM Head 使用输入末位置的特征预测 `40`。
4. 追加 `40` 时，其 KV 还没计算，所以总长是 4、已缓存数是 3。
5. Decode 输入 `40`，读取前面的历史并写入 `40` 的 KV，预测 `50`。
6. 达到两个新增 token，释放请求的块引用。缓存计数清零、块表清空，但 `[40,50]` 仍能作为结果返回。

### 4.2 Chunked Prefill：输入切片不是输出切片

假设输入 9 个 token，预算为 3，没有前缀命中：

```text
第1轮 Prefill：输入位置 [0,3)，缓存数 0→3，候选丢弃
第2轮 Prefill：输入位置 [3,6)，缓存数 3→6，候选丢弃
第3轮 Prefill：输入位置 [6,9)，缓存数 6→9，候选追加为第1个 completion token
下一轮 Decode：输入刚追加的尾 token，继续生成
```

后续 Prefill 切片的 Q 仅包含新输入，但 K/V 历史包括先前切片。切片减少本轮新输入计算量，**不等于此版本只为切片分配 KV 块**：`BlockManager.allocate` 为当前完整序列预留块表。

### 4.3 Prefix Cache：省去历史 token 的重算，不省去对历史 KV 的读取

默认块大小为 256。假设请求有 300 个输入 token，首个整块命中：

```text
命中前缀：256 token 的 KV 已存在
本轮新输入：剩余 44 token
Q 长度：44
K 长度：300
```

`can_allocate` 只检查最后一个逻辑块之前的连续前缀；遇到首个 miss 就停止。哈希链包含父前缀信息，命中还检查当前块 token 列表。共享 used 块增加引用；命中 free 块则重新激活。

`deallocate` 不马上擦掉 KV，也不删除全部前缀哈希。因此 free 块仍能命中；只有被 `_allocate_block` 改用于新内容时，旧映射才失效。

### 4.4 抢占：丢 KV 引用，不丢已生成 token

Decode 跨块边界且无空闲块时，Scheduler 优先从剩余 running 队尾选择牺牲请求：

```text
被抢占请求：RUNNING → WAITING
保持 token_ids、num_prompt_tokens、生成策略
清空 block_table、num_cached_tokens=0、is_prefill=True
放到 waiting 队头
后续从“prompt + 已生成 token”恢复 Prefill
```

这不是 GPU kernel 执行中途的暂停，也不是把 KV 交换到 CPU。恢复时可能命中仍未覆盖的前缀，但不能保证全部旧 KV 可复用。

调度器没有显式优先级打分。waiting 请求容量不足时不能反过来抢占 running 以强行进入 Prefill；它只能等待之后再次检查。队尾请求仍可能反复被抢占，没有完整的公平性保护。

## 5. 核心变量对照表

### 5.1 形状字母

| 符号 | 含义 |
|---|---|
| B | 本轮请求数量 |
| T | 本轮实际输入的 query token 总数；纯 Decode 通常 T=B |
| H | hidden_size，每个 token 主干特征的宽度 |
| V | vocab_size，候选 token 的总数；不是 Attention 中的 Value |
| P | tensor_parallel_size / world_size，TP rank 数 |
| Nq / Nkv | 每卡 Q 头数 / KV 头数 |
| Dh | head_dim，每个头的特征宽度 |
| I | intermediate_size，MLP 全局中间宽度 |
| L / Nb / S | 层数 / GPU 池物理块数 / 每块 token 槽数量 |

### 5.2 请求与块管理：CPU 状态

| 名称 | 单位/内容 | 谁写，谁读 | 常见误解 |
|---|---|---|---|
| `seq_id` | 进程内递增整数 | Sequence 创建，Engine 排序 | 不是优先级分数 |
| `token_ids` | CPU 整数列表，prompt + completion | Sequence/Scheduler 写，Runner/Engine 读 | 不是 KV Tensor |
| `num_prompt_tokens` | 原始输入 token 数 | 创建时固定 | 不包含重计算时已有 completion |
| `num_tokens` | 当前总 token 数 | append_token 增加 | 不等于已缓存数量 |
| `num_cached_tokens` | 具有有效 KV 的连续前缀 token 数 | 块管理和 postprocess 写，Runner 读 | 生成一个新 ID 不会立刻加一 |
| `num_scheduled_tokens` | 本轮安排计算的输入 token 数 | schedule 写，Runner/postprocess 读 | 不是本轮生成文字长度 |
| `Sequence.is_prefill` | 序列快照传输模式 | 创建/抢占/Decode 选择时写 | 不同于整个 batch 的 is_prefill 参数 |
| `block_table` | 物理 block ID 的 CPU 列表 | BlockManager 写，Runner 读 | 元素不是 GPU 字节地址 |
| `Block.ref_count` | 持有该块的活跃引用数 | BlockManager | 减一不一定变空闲 |
| `free_block_ids` | 无活跃引用、可重用的 ID 队列 | BlockManager | free 不意味着旧 KV 已清零 |
| `hash_to_block_id` | 前缀哈希 → 一个物理块 ID | hash_blocks / 分配失效逻辑 | 不是缓存数值本身，也不只索引 used |

### 5.3 输入、元数据和计算：GPU Tensor

| 名称 | 形状与 dtype | 用途 |
|---|---|---|
| `input_ids` | `[T]`，int64 | 本轮真正输入的 token ID |
| `positions` | `[T]`，int64 | 每个输入 token 在所属请求中的逻辑位置，供 RoPE |
| `cu_seqlens_q` | Prefill `[B+1]`，int32 | 变长新 query 的累积边界 |
| `cu_seqlens_k` | Prefill `[B+1]`，int32 | 各请求历史 + 新输入的 key 长度累积值 |
| `context_lens` | Decode `[B]`，int32 | 各请求参与 Attention 的长度，包含本轮尾 token |
| `slot_mapping` | `[T]`，int32 | 本轮各 token 的 KV 写入槽位；-1 跳过 |
| `block_tables` | `[B,max_blocks]`，int32 | 补齐后的逻辑→物理块表，读历史 KV |
| `hidden_states` | 主干 `[T,H]` | 逐位置模型特征；输出头选择后为 `[B,H]` |
| `q` | `[T,Nq,Dh]` | 本轮 query；Decode 调接口时增加长度为 1 的轴 |
| `k / v` | 新输入 `[T,Nkv,Dh]` | 本轮新计算 K/V，随后按槽位写入持久缓存 |
| `kv_cache` | `[2,L,Nb,S,Nkv,Dh]`，模型 dtype | 持久 GPU KV 池，各层通过视图访问 |
| `logits` | rank 0 `[B,V]`；每卡局部 `[B,V/P]` | 下一 token 候选分数，不是概率 |
| `temperatures` | `[B]`，float32 | 逐请求温度；采样时扩为 `[B,1]` |
| 采样 `token_ids` | GPU 整型 `[B]` → CPU list[int] | 每条请求一个候选，由 Scheduler 写回 |

`Context` 本身是 Python 对象，但里面的 Tensor 指向 GPU 数据；“元数据”不等于“全部位于 CPU”。

## 6. 三段值得手算的公式

### 6.1 槽位：句子位置与物理存放位置不同

```text
S=4，block_table=[5,2]
逻辑位置 0..5 → 物理槽 [20,21,22,23,8,9]

slot = block_table[position // S] * S + position % S
```

这是概念公式。实际代码在 `prepare_prefill` 按块展开区间，Decode 用最后块 ID 和块内尾偏移，结果按本轮输入顺序排列。

### 6.2 末行：每条请求只计算一个续写位置的词表分数

```text
cu_seqlens_q=[0,3,5]
[1:] 去掉第一个边界 → [3,5]
逐元素减 1 → [2,4]
```

`ParallelLMHead.forward` 选特征第 2、4 行，得到 `[B,H]` 后投影为 `[B,V]`。不是前面的 token 不计算主干，而是它们不需要在这轮各自输出整个词表的 logits。

### 6.3 显存：块字节数包含所有层及 K/V

```text
block_bytes = 2 * L * S * Nkv * Dh * dtype.itemsize
```

2 代表 K 和 V；`itemsize` 为每个数占的字节数。每卡 KV 头数已按 P 切分，不能再重复除以 P。

## 7. 多卡与 CUDA Graph 的阅读边界

### 7.1 控制通道与数据通道

| 通道 | 在哪里 | 传输/操作什么 |
|---|---|---|
| spawn | LLMEngine.__init__ | 创建 worker Python 进程 |
| SharedMemory + Event | ModelRunner.write_shm/read_shm/loop | 方法名称与参数快照；通知 worker 开始 |
| NCCL all_reduce | RowParallelLinear、VocabParallelEmbedding | 对 GPU 局部结果求和，合成完整输出 |
| NCCL gather | ParallelLMHead | rank 0 收集各词表区间的 logits 再拼接 |
| NCCL barrier | Runner 控制通道初始化/退出 | 协调各 rank 的资源创建/销毁时序 |

各 rank 必须按兼容顺序进入通信；只有 rank 0 调用主 Scheduler 和 Sampler。worker 的 Sequence 快照没有主对象全部属性。

### 7.2 图捕获与重放

1. 初始化时为几种 Decode 容量建立图，输入/长度/槽位/块表/输出缓冲区地址固定。
2. 每轮选择容量不小于实际 B 的图，将新值复制进这些缓冲区。
3. 多余行的 `slot_mapping=-1`、`context_lens=0`，避免无效缓存写入并约束读取。
4. 重放模型主干，截取有效输出，再在图外执行 LM Head 和采样。

`enforce_eager=True` 跳过图，但不移除计算层上的 `@torch.compile`。Prefill 也走普通主干调用。图不是保存上一次答案，也不是自动支持任意动态 batch 或混合阶段。

## 8. 当前实现的能力与潜在问题

下面是依据源码可以确认的边界，不是对作者设计意图的推测。本次仅记录，没有修复。

| 项目 | 具体依据 | 影响或后续检查方向 |
|---|---|---|
| 支持切片但不支持 P/D 混批 | Scheduler.schedule 有 Prefill 提前返回；Context 为单个 is_prefill | 切片可控制单步输入量，但不能自动与 Decode 一起计算 |
| 切片仍预留完整输入块 | BlockManager.allocate 遍历至 seq.num_blocks | 不应声称 chunk size 同比例降低持久 KV 占用 |
| 等待/抢占公平性有限 | 队头选择、队尾抢占、已选请求回队头 | 队尾可能反复被抢占；容量不足也不保证总能前进 |
| 输入/策略校验有限 | Sequence 直接取 token_ids[-1]；add_request 未完整检查长度；postprocess 用 == 检查上限 | 空输入、非正 max_tokens、超长请求、策略列表长度不一致需要调用者避免 |
| generate 返回标注与实际不符 | 标注 list[str]，实际是字典列表；run 的 worker 返回 None | 只在 docstring 说明，未改接口/类型标注 |
| 部分 CUDA Graph 容量边界未覆盖 | graph_bs 固定含 1/2/4/8，之后按 16 步进 | max_num_seqs 小于 8 或为某些非 16 倍数时，需检查缓冲区和档位；如上限 20 不含 20/32 |
| RMSNorm 的 FP32 别名 | float()/to(orig_dtype) 在 dtype 相同时可复用存储，并有 mul_/add_ | 不可无条件将 residual 当作独立、未归一化的快照 |
| 固定通信资源与退出约定 | localhost:2333、共享内存名 nanovllm、固定 1 MiB；exit 删除 Runner | 同机多引擎可能冲突，大控制消息需检查；手动 exit 后自动退出重复调用也需注意 |
| 并发隔离有限 | Context 是进程全局变量，Sequence.block_size 是类属性，get_rope 的缓存键不含设备 | 同进程多引擎/并发 forward/跨设备模块复用不是已验证能力 |
| 缓存清理的异常路径有限 | run 正常末尾 reset_context，没有 try/finally | 异常时可能遗留本轮 Context；本次没有加入异常处理 |
| 模型/权重兼容性范围有限 | 固定创建 Qwen3；RoPE scaling 仅读 rope_theta；loader 未检查完整参数清单 | 不应把“读到 AutoConfig”理解为已支持任意模型或所有 RoPE 变体 |
| TP 分片约束 | Q/KV 头数、词表/投影宽度要求整除 P | 不是支持任意 TP 数，也没有 KV 头复制分片模式 |

哈希不是形式化的无碰撞证明；代码核对当前块 token，但没有存储完整父 token 链来逐项验证父哈希碰撞。通常按哈希前缀机制理解，不应声称绝对不可能碰撞。

## 9. 如何确认这次没有改变逻辑

验收采用以下方法：

1. 对全部 21 个 Python 文件执行内存语法编译，不靠导入整个 CUDA 引擎完成语法检查。
2. 对比修改前后 AST：去除新增 docstring 后，全部可执行结构应相同；原有源码行必须按原顺序保留。
3. 检查所有类/函数都具有 docstring，所有核心文件均有职责说明。
4. 运行 CPU 最小测试：分块进度、中间采样丢弃、前缀命中/共享/失效、引用回收、抢占恢复。
5. 在可用 CUDA/已下载权重下运行小规模单卡冒烟测试，不将它等同于性能评测或多卡验证。
6. 运行 `git diff --check`，核对指南中的本地链接和函数名称。

新增 docstring 会改变 Python 对象的 `__doc__`，也会改变源码行号；这些属于教学说明变化，不是执行算法的修改。编译/图缓存可能重新构建，不能据此直接判断吞吐变化。

仓库当前没有发现独立的 pytest 测试目录或 CI 检查配置。因此本次执行的是针对源码的额外检查，不声称通过一个不存在的完整测试套件。单卡通过也不代表 NCCL 多卡的数值与故障路径全部验证。

### 本次实际检查结果

| 检查 | 结果与范围 |
|---|---|
| 语法与 AST | 全部 21 个 Python 文件通过语法编译；去除 docstring 后 AST 与基线完全一致；所有原有源码行按原顺序保留 |
| 注释覆盖 | 29 个类、108 个函数/方法全部有 docstring，核心文件有文件级说明；复杂调度、映射、缓存和图逻辑有步骤注释 |
| CPU 状态测试 | 分块 Prefill、丢弃中间候选、Decode/终止、前缀共享及空闲复用、旧哈希失效、抢占与部分前缀恢复通过 |
| GPU 槽位写入 | Triton 正确写入指定物理槽；`slot=-1` 跳过写入的最小测试通过 |
| 单卡生成 | WSL Python 3.12.15、PyTorch 2.8.0+cu128、RTX 4090 Laptop GPU；本地 Qwen3-0.6B 完成两条相同的 300-token 输入，各生成 3 个 token |
| 单卡执行路径 | `max_num_batched_tokens=64`、`max_num_seqs=16`、`max_model_len=512`、`enforce_eager=False`；经历分块输入、前缀复用、Decode 图捕获/重放，结束后所有活跃块引用归零，资源退出成功 |
| 文档/差异 | 阅读指南的 26 个本地链接及 README 的本地链接有效，代码块围栏配对；Windows Git 的 `diff --check` 通过 |
| 多卡/性能 | 仅一张 GPU，未运行多卡 TP 验证；没有做正式吞吐、TTFT、ITL 或结果质量评测 |

GPU 检查出现第三方 PyTorch/Triton 的逻辑运算弃用警告及 `_POSIX_C_SOURCE` 宏重复定义编译警告，但检查完成且断言通过；本次未修改依赖或压制警告。

## 10. 读完后的复述练习

不看文档，用自己的话回答：

1. 一条输入怎样从字符串变成 Sequence，再变成 GPU Tensor？
2. 哪个函数决定“本轮计算多少输入 token”？哪个函数真正写入 KV？
3. 已生成一个新 token 后，为什么 token 总数可以大于已缓存数？
4. 前缀命中时，哪些计算省掉，哪些历史读取仍然需要？
5. 被抢占请求的哪些东西保留、哪些被清空？恢复时从哪里继续？
6. 为什么 `cu_seqlens_q=[0,3,5]` 能找到末行 `[2,4]`？
7. Embedding、LM Head、Sampler、Tokenizer 各负责哪一步？
8. 两张卡的列并行输出为什么不一定立即收集，而行并行要求和？
9. CUDA Graph 的固定地址与每轮变化的 token/长度如何兼容？
10. 当前实现与“在线服务、混合 P/D、多模态、任意模型兼容”之间还缺什么？

能把这十项串成一次完整请求的故事，就已经从“认识类名”走到了“理解数据与状态怎样流动”。
