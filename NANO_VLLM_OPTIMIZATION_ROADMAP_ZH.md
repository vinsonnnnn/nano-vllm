# Nano-vLLM 未来优化路线图

> 面向当前仓库 `nano-vllm 0.2.0` 的性能、工程质量与功能演进建议。
> 编写日期：2026-09-21。
> 本文是一份设计与实施路线图，不代表其中所有能力已经实现。

---

## 目录

- [1. 总体结论](#1-总体结论)
- [2. 优化原则与项目定位](#2-优化原则与项目定位)
- [3. 当前架构与已具备的能力](#3-当前架构与已具备的能力)
- [4. 当前最值得关注的瓶颈](#4-当前最值得关注的瓶颈)
- [5. 优先级总表](#5-优先级总表)
- [6. P0：先建立正确性与性能基线](#6-p0先建立正确性与性能基线)
- [7. P1：改进 Prefill 与 Decode 调度](#7-p1改进-prefill-与-decode-调度)
- [8. P1：消除 Chunked Prefill 的无效采样](#8-p1消除-chunked-prefill-的无效采样)
- [9. P1：复用输入和元数据缓冲区](#9-p1复用输入和元数据缓冲区)
- [10. P1：优化采样器](#10-p1优化采样器)
- [11. P1：完善输入校验与资源生命周期](#11-p1完善输入校验与资源生命周期)
- [12. P2：优化 KV Cache 与前缀缓存](#12-p2优化-kv-cache-与前缀缓存)
- [13. P2：优化 CUDA Graph 与编译策略](#13-p2优化-cuda-graph-与编译策略)
- [14. P2：优化张量并行与进程通信](#14-p2优化张量并行与进程通信)
- [15. P2：分片加载模型权重](#15-p2分片加载模型权重)
- [16. P2：量化与 Kernel 优化](#16-p2量化与-kernel-优化)
- [17. P3：投机解码与高级推理能力](#17-p3投机解码与高级推理能力)
- [18. P3：从离线引擎走向在线服务](#18-p3从离线引擎走向在线服务)
- [19. Benchmark 设计](#19-benchmark-设计)
- [20. 测试策略](#20-测试策略)
- [21. 推荐实施里程碑](#21-推荐实施里程碑)
- [22. 推荐的 PR 拆分方式](#22-推荐的-pr-拆分方式)
- [23. 不建议过早进行的优化](#23-不建议过早进行的优化)
- [24. 最推荐优先落地的三个改动](#24-最推荐优先落地的三个改动)
- [25. 参考资料](#25-参考资料)

---

## 1. 总体结论

Nano-vLLM 已经包含现代 LLM 推理引擎最重要的骨架：

- Continuous Batching；
- Chunked Prefill；
- Paged KV Cache；
- Prefix Caching；
- FlashAttention；
- Tensor Parallelism；
- `torch.compile`；
- CUDA Graph。

下一阶段最有价值的工作不是马上堆叠更多模型和功能，而是把现有链路做得更加可测量、低延迟和稳定。

推荐主线为：

```text
建立基线与测试
    ↓
消除确定存在的重复计算
    ↓
降低每个 Decode step 的 CPU 准备开销
    ↓
重新设计 Prefill/Decode 调度公平性
    ↓
优化 KV Cache、TP 通信和模型加载
    ↓
最后再做量化、投机解码和在线服务
```

如果只选择一个最能体现推理引擎价值的长期方向，建议选择：

> 设计统一 token budget 的 Scheduler，让 Decode 和 Chunked Prefill 得到合理调度，并使用 TTFT、TPOT、P99 延迟和吞吐量共同评价结果。

---

## 2. 优化原则与项目定位

### 2.1 保持“小而可读”

Nano-vLLM 的核心优势是代码量小、结构清楚。一个优化即使能提高性能，如果引入大量难以维护的抽象，也可能损害项目最重要的教学价值。

建议每项优化都回答四个问题：

1. 它解决了哪个可测量的瓶颈？
2. 性能收益是否覆盖代码复杂度？
3. 是否能通过开关关闭，并保留容易调试的 eager 路径？
4. 是否有正确性测试和性能回归测试？

### 2.2 正确性优先于吞吐量

推理引擎的错误可能不是立即崩溃，而是悄悄改变输出概率、污染 KV Cache，或者只在 Prefix Cache、TP、CUDA Graph 组合下出现。

必须优先验证：

- eager 与 CUDA Graph 输出一致；
- Prefix Cache 命中与不命中输出一致；
- Chunked Prefill 与完整 Prefill 输出一致；
- TP=1 与 TP>1 的 logits/采样结果在允许误差内一致；
- 抢占并重算后输出正确；
- 优化采样器后概率分布不变。

### 2.3 同时衡量吞吐和延迟

只看 output tokens/s 容易得到错误结论。例如不断插入长 Prefill 可以提高 GPU 利用率，却让正在 Decode 的用户等待更久。

至少同时关注：

```text
吞吐量              output tokens/s、prompt tokens/s
首 token 延迟       TTFT
单 token 延迟       TPOT、ITL
请求完成延迟        E2E latency
尾延迟              P95、P99
显存效率            KV capacity、并发量
缓存效率            prefix cache hit rate
重算成本            preemptions、recomputed tokens
```

### 2.4 区分离线吞吐与在线延迟

当前公开 API 是同步离线 `generate()`。优化目标需要先明确：

- 离线批量推理：优先整体吞吐；
- 在线对话服务：优先 TTFT、TPOT、公平性和取消能力；
- 长上下文：优先 KV Cache 容量和 Prefill 效率；
- 多租户共享前缀：优先 Prefix Cache 命中和安全隔离。

不能用一个配置同时声称对所有场景最佳。

---

## 3. 当前架构与已具备的能力

当前调用链为：

```text
LLM.generate
  -> Sequence
  -> Scheduler
  -> BlockManager
  -> ModelRunner
  -> Qwen3 model
  -> LM Head
  -> Sampler
  -> Scheduler.postprocess
```

当前实现已经做了很多正确且有价值的优化：

1. Prefill 使用变长 FlashAttention；
2. Decode 使用分页 KV Cache；
3. 用 Triton kernel 写入 K/V；
4. 完整 block 支持 Prefix Cache；
5. Decode 支持 CUDA Graph；
6. QKV 与 Gate/Up 权重合并；
7. RMSNorm、RoPE、激活和采样使用 `torch.compile`；
8. Decode 时只向 TP 子进程序列化最后一个 token；
9. 显存不足时支持 recompute preemption；
10. 长 Prompt 支持 Chunked Prefill。

因此，未来优化应尽量围绕这些现有设计迭代，而不是全部推倒重写。

---

## 4. 当前最值得关注的瓶颈

对应的主要源码热点：

| 关注点 | 当前源码 |
|---|---|
| Prefill/Decode 调度与抢占 | [`scheduler.py`](nanovllm/engine/scheduler.py#L25-L92) |
| 输入和 Attention 元数据准备 | [`model_runner.py`](nanovllm/engine/model_runner.py#L123-L193) |
| 模型、LM Head 与采样调用 | [`model_runner.py`](nanovllm/engine/model_runner.py#L195-L220) |
| CUDA Graph bucket 与捕获 | [`model_runner.py`](nanovllm/engine/model_runner.py#L223-L257) |
| 温度采样 | [`sampler.py`](nanovllm/layers/sampler.py#L7-L12) |
| Prefix Cache 与 block 生命周期 | [`block_manager.py`](nanovllm/engine/block_manager.py#L35-L120) |
| TP 词表 logits gather | [`embed_head.py`](nanovllm/layers/embed_head.py#L56-L65) |
| Safetensors 权重加载 | [`loader.py`](nanovllm/utils/loader.py#L12-L28) |
| API、配置过滤和生成循环 | [`llm_engine.py`](nanovllm/engine/llm_engine.py#L17-L90) |

### 4.1 Prefill 严格优先于 Decode

`Scheduler.schedule()` 只要调度到任何 Prefill，就立即返回 Prefill batch。Decode 只有在本轮没有 Prefill 时才会运行。

影响：

- 大量等待请求可能延迟正在运行请求的下一个 token；
- 在线负载下可能产生较高 ITL 和尾延迟；
- 当前策略偏离线吞吐，不适合所有服务场景。

### 4.2 Chunked Prefill 的中间块仍计算 logits 并采样

`ModelRunner.run()` 总会执行 LM Head 和 Sampler，但 `Scheduler.postprocess()` 会丢弃尚未完成 Prefill 的临时 token。

影响：

- 产生确定无用的词表投影；
- 产生确定无用的采样；
- TP 模式下还可能产生无用的完整 logits gather。

### 4.3 每个 step 重复构造 CPU 列表和 Tensor

Prefill/Decode 每轮都重新创建：

- `input_ids`；
- `positions`；
- `slot_mapping`；
- `context_lens`；
- `cu_seqlens_q/k`；
- padded `block_tables`；
- pinned CPU Tensor；
- GPU Tensor。

对小模型和小 Decode batch，这些 Python 与内存管理开销可能占据明显比例。

### 4.4 Sampler 进行完整词表 Softmax

当前 exponential-race 采样先计算完整 Softmax，再创建同等大小的指数随机 Tensor。Softmax 的公共归一化因子实际上不会影响最终 argmax。

### 4.5 TP rank 0 收集完整词表 logits

每个 rank 计算一部分词表 logits 后，rank 0 gather 并拼接全部 logits。随着 batch 和词表增大，通信和显存临时空间都会增长。

### 4.6 每个 TP rank 读取完整 checkpoint Tensor

权重加载时先 `get_tensor()` 读取完整 Tensor，再由每个 rank 的 loader 切片。多卡时产生重复 I/O 和 CPU 临时内存。

### 4.7 Prefix Cache 粒度较粗

当前物理 block size 最小为 256 token，短公共前缀无法命中。Block hash 还会反复把 Python list 转成 NumPy 数组。

### 4.8 CUDA Graph 捕获数量固定

默认最多捕获 36 个 batch size：

```text
1, 2, 4, 8, 16, 32, 48, ..., 512
```

这在启动时间、Graph 内存和 padding 计算之间做了固定取舍，但没有根据真实流量自动调整。

### 4.9 输入校验和生命周期管理较弱

例如：

- 未知 `LLM` 配置参数会被静默忽略；
- prompts 与 SamplingParams 列表长度不一致时，`zip` 静默截断；
- 空 token 列表会在 `token_ids[-1]` 失败；
- 缺少 prompt + output 长度的入口校验；
- 多卡端口和共享内存名称固定；
- 异常退出时子进程和共享内存清理能力有限。

这些问题不一定降低正常 benchmark 吞吐，却会影响长期优化的可信度。

---

## 5. 优先级总表

| 优先级 | 方向 | 主要目标 | 预期收益 | 实现难度 | 主要风险 |
|---|---|---|---|---|---|
| P0 | 指标、Profiler、测试 | 建立可信基线 | 间接但必要 | 低～中 | 测试负担增加 |
| P1 | 调度策略 | 降低 TTFT/TPOT/P99 | 高 | 中～高 | 吞吐与公平性权衡 |
| P1 | 跳过中间 Prefill 采样 | 消除无效工作 | 中 | 低 | 混合批次采样 mask |
| P1 | 元数据 buffer 复用 | 降低 CPU/H2D 开销 | 中～高 | 中 | buffer 生命周期 |
| P1 | Sampler 融合 | 降低词表读写 | 中 | 中 | 随机分布正确性 |
| P1 | 校验与清理 | 提升稳定性 | 中 | 低～中 | API 行为变化 |
| P2 | Prefix/KV Cache | 提升命中和并发 | 高 | 高 | kernel 与精度约束 |
| P2 | CUDA Graph 策略 | 降低启动和 Decode 开销 | 中 | 中～高 | Graph 安全性 |
| P2 | TP 通信 | 提升多卡吞吐 | 高 | 高 | 分布式正确性 |
| P2 | 分片权重加载 | 降低启动内存和 I/O | 中 | 中 | packed 权重切片复杂 |
| P2 | 权重/KV 量化 | 提升容量和吞吐 | 高 | 高 | 精度与硬件兼容 |
| P3 | 投机解码 | 减少主模型 Decode 次数 | 高 | 很高 | 接受率和调度复杂度 |
| P3 | 在线服务 | 支持真实请求流量 | 功能价值高 | 高 | 并发、取消、容错 |

表中的预期收益是源码分析判断，必须通过第 19 节的 Benchmark 实测确认。

---

## 6. P0：先建立正确性与性能基线

### 6.1 增加核心指标

建议为每条请求记录时间点：

```text
ARRIVED              请求进入引擎
FIRST_SCHEDULED      第一次进入 Scheduler 输出
PREFILL_FINISHED     Prompt 的 KV 全部计算完成
FIRST_TOKEN          第一个 completion token 产生
TOKEN_i              每个后续 token 产生
FINISHED             EOS 或 max_tokens
PREEMPTED            被抢占
RESUMED              被重新调度
```

由此计算：

```text
TTFT = FIRST_TOKEN - ARRIVED
E2E  = FINISHED - ARRIVED
TPOT = (FINISHED - FIRST_TOKEN) / (output_tokens - 1)
ITL  = TOKEN_i - TOKEN_(i-1)
Queue Time = FIRST_SCHEDULED - ARRIVED
```

引擎级指标：

- prompt tokens/s；
- output tokens/s；
- waiting/running 数量；
- KV Cache 使用率；
- Prefix Cache query/hit tokens；
- block 分配与回收数量；
- preemption 次数；
- recomputed tokens；
- 每一步的 Prefill/Decode batch size；
- 每一步 CPU prepare、H2D、model、LM Head、sampling 时间。

### 6.2 增加阶段计时

普通 `perf_counter()` 只能可靠衡量 CPU 路径。GPU 异步操作应使用：

- CUDA Event；
- PyTorch Profiler；
- NVTX range；
- Nsight Systems；
- Nsight Compute。

建议给以下区域加入 NVTX：

```text
scheduler
prepare_inputs
h2d_copy
model_forward
lm_head
sampler
postprocess
tp_control_ipc
```

### 6.3 建立可重复输入集

至少准备四类 workload：

1. 短输入短输出：32/32；
2. 短输入长输出：32/512；
3. 长输入短输出：2048/32；
4. 长输入长输出：2048/512。

再组合：

- 并发 1、8、32、128、512；
- 相同前缀与完全不同前缀；
- eager 与 CUDA Graph；
- TP=1、2、4；
- 固定长度与真实分布。

### 6.4 建立正确性基线

建议增加一个 deterministic greedy 模式，并与 Hugging Face Transformers 对比：

- 单 token logits；
- 完整 Prefill 输出；
- 多轮 Decode 输出；
- Chunked Prefill；
- Prefix Cache；
- 抢占重算；
- CUDA Graph；
- TP。

随机采样无法直接逐 token 对比，所以 greedy 是重要的测试工具，而不只是用户功能。

### 6.5 完成标准

- 每次优化 PR 都能输出优化前后同一 workload 的指标；
- CI 中有不依赖 GPU 的 Scheduler/BlockManager 测试；
- GPU 环境中有最小端到端正确性测试；
- 性能结果记录硬件、驱动、CUDA、PyTorch、FlashAttention、模型和配置。

---

## 7. P1：改进 Prefill 与 Decode 调度

### 7.1 当前问题

当前调度器把每一轮整体分成 Prefill 或 Decode：

```text
只要 waiting 中有可调度的 Prefill
    -> 本轮只做 Prefill
否则
    -> 本轮做 Decode
```

这对离线吞吐简单有效，但无法精细控制正在生成请求的 token 间延迟。

### 7.2 第一阶段：Decode 优先策略

不立刻重构 Attention batch 的情况下，可以：

1. 先从 running 选择 Decode batch；
2. 如果没有 running，执行 Prefill；
3. 或限制连续 Prefill step 数；
4. 给等待过久的 Prefill 提升优先级；
5. 添加策略开关：`throughput`、`latency`、`balanced`。

示意：

```python
if running and should_serve_decode():
    return schedule_decode()
return schedule_prefill()
```

这种方案一次仍只执行一种 Attention 路径，代码改动较小。

### 7.3 第二阶段：统一 token budget

把每条请求统一表示为：

```text
num_tokens_total      Prompt + 已生成 + draft token
num_computed_tokens   已经拥有有效 KV 的 token 数
remaining_compute     total - computed
```

Scheduler 不再依赖全局 Prefill/Decode 二分，而是决定每个请求本轮计算多少 token：

```text
Decode 请求通常需要 1 token
Chunked Prefill 请求需要 1..N token
Prefix Cache 命中会直接增加 computed tokens
抢占后 computed tokens 回退并重新追赶
```

### 7.4 混合执行的两种实现

方案 A：一个 Scheduler step 内执行两个子 batch。

```text
schedule
  -> Decode sub-batch -> forward
  -> Prefill sub-batch -> forward
  -> 合并 postprocess
```

优点：复用现有两条 Attention 路径；缺点：一次调度产生两次 forward。

方案 B：真正混合 Prefill/Decode。

需要：

- 每条序列自己的 query/context 元数据；
- Attention backend 支持混合 batch；
- LM Head 只选择需要采样的位置；
- CUDA Graph 适配混合形状。

优点：更灵活；缺点：复杂度明显增加。

Nano-vLLM 更适合先实现方案 A，再根据 Profiler 判断是否值得做方案 B。

### 7.5 公平性与抢占

建议加入：

- 每请求等待 step 数；
- aging，等待越久优先级越高；
- 最大连续 Prefill token 数；
- 最大连续 Decode step 数；
- 抢占成本估计：优先抢占重算代价较低的请求；
- 已生成较多 token 的请求适当提高完成优先级；
- cache-aware scheduling：优先处理高 Prefix Cache 命中请求。

### 7.6 验收指标

- P99 ITL 显著下降；
- TTFT 不出现不可接受退化；
- 总吞吐下降在设定预算内；
- 无请求长期饥饿；
- preemption 和 recomputed tokens 不增加或可解释；
- 不同调度策略的取舍有 benchmark 数据。

---

## 8. P1：消除 Chunked Prefill 的无效采样

### 8.1 当前流程

中间 Prefill chunk 当前仍然执行：

```text
Transformer
-> LM Head
-> Sampler
-> Scheduler 发现 Prefill 未结束
-> 丢弃 token
```

### 8.2 推荐设计

让 Scheduler 输出明确的采样信息：

```python
ScheduleOutput(
    seqs=...,
    is_prefill=...,
    sample_mask=...,
)
```

当前调度策略下，部分 Prefill chunk 通常独占 batch，可以先实现 batch 级 `should_sample=False`。为未来混合调度，应最终支持每请求 `sample_mask`。

运行路径：

```text
should_sample=False
    -> 只运行 model forward 并写 KV
    -> 不运行 LM Head
    -> 不准备 temperature
    -> 不运行 Sampler
    -> 返回空 token 结果
```

### 8.3 注意事项

- `num_cached_tokens` 仍然必须正确增加；
- 完整 block 仍需建立 Prefix Cache hash；
- 最后一个 Prefill chunk 必须采样；
- TP 所有 rank 必须遵循相同控制流；
- 未来一个 batch 同时含中间 chunk 和结束 chunk 时，需要只对结束位置计算 logits。

### 8.4 验收指标

- Chunked 与非 Chunked 输出完全一致；
- 中间 chunk 的 Profiler 中不再出现 LM Head 和 Sampler；
- 长 Prompt Prefill 总时间不增加；
- TP 模式不再为中间 chunk gather logits。

---

## 9. P1：复用输入和元数据缓冲区

### 9.1 优化目标

减少每个 step 中：

- Python 对象创建；
- list append/extend；
- pinned memory 分配；
- GPU Tensor 分配；
- 重复 block table padding；
- 同步 H2D 拷贝等待。

### 9.2 预分配 Buffer

在 `ModelRunner` 初始化时，根据上限创建：

```text
CPU pinned:
  input_ids_cpu[max_num_batched_tokens]
  positions_cpu[max_num_batched_tokens]
  slot_mapping_cpu[max_num_batched_tokens]
  cu_seqlens_cpu[max_num_seqs + 1]
  context_lens_cpu[max_num_seqs]
  block_tables_cpu[max_num_seqs, max_num_blocks]
  temperatures_cpu[max_num_seqs]

GPU:
  对应的长期存在 Tensor
```

每轮只写 `[:valid_length]`，避免重新创建 Tensor。

### 9.3 Persistent Batch

Decode 中很多 running 序列连续多轮不变。可以维护 persistent batch：

- 新增请求只插入空 slot；
- 完成请求只标记 slot 空闲；
- block table 只在新增 block、抢占或请求切换时更新；
- 每轮主要更新 last token、position、context length。

这与 CUDA Graph 的固定地址和固定容量特别匹配。

### 9.4 拷贝与计算重叠

可以设计双缓冲：

```text
GPU 正在运行 batch N
CPU 同时准备 batch N+1
独立 CUDA stream 复制 batch N+1
Event 保证使用前完成
```

需要防止 Scheduler 修改仍被 GPU 使用的元数据。

### 9.5 GPU 端生成简单元数据

Decode 的部分元数据有简单公式：

```text
position = context_len - 1
slot = last_block_id * block_size + last_block_num_tokens - 1
```

可以评估用一个小 GPU kernel 从紧凑输入生成，减少 CPU 循环和 H2D 字节数。是否值得必须依据小 batch 和大 batch的 Profiler。

### 9.6 验收指标

- Decode step 的 CPU prepare 时间下降；
- `torch.cuda.memory_allocated()` 在稳态不因 step 持续波动；
- 小 batch TPOT 改善；
- CUDA Graph replay 前的 host gap 缩短；
- 连续数千 step 无 buffer 覆盖或脏数据问题。

---

## 10. P1：优化采样器

### 10.1 去掉不必要的 Softmax

当前算法为：

```text
z = logits / temperature
p = softmax(z)
token = argmax(p / E), E ~ Exponential(1)
```

由于 Softmax 分母对同一行所有 token 相同：

```text
argmax(softmax(z) / E)
= argmax(exp(z) / E)
= argmax(z - log(E))
```

因此可以省去 Softmax：

```text
score = logits / temperature - log(E)
token = argmax(score)
```

### 10.2 融合 Kernel

理想的 Triton Sampler 可以在一个逻辑流程中完成：

1. 读取 logits；
2. temperature scaling；
3. 生成或读取随机数；
4. 计算 Gumbel/exponential score；
5. block 内 reduction；
6. 输出 token ID。

目标是避免创建完整 `probs` 和多个词表大小临时 Tensor。

### 10.3 增加 Greedy

`temperature=0` 应明确走：

```python
token = logits.argmax(dim=-1)
```

Greedy 的价值包括：

- 用户可重复输出；
- 与 Transformers 做确定性正确性对比；
- 更容易验证 Prefix Cache、TP 和 CUDA Graph；
- 不需要随机数和 Softmax。

### 10.4 Top-k 与 Top-p

推荐顺序：

1. greedy；
2. temperature sampling；
3. top-k；
4. top-p；
5. repetition/frequency penalty。

Top-k 更容易并行实现；Top-p 需要排序或近似筛选，复杂度更高。

### 10.5 RNG 设计

应支持：

- 全局 seed；
- 每请求 seed；
- CUDA Graph replay 下正确推进 RNG state；
- 请求调度顺序变化不意外改变其他请求随机流；
- TP 模式各 rank 的随机数不重复覆盖词表区域。

### 10.6 验收指标

- 固定 seed 可复现；
- greedy 与参考实现一致；
- 随机采样经过统计分布测试；
- Profiler 中不再出现完整词表 Softmax；
- 大词表和大 batch 下采样时间、临时显存下降。

---

## 11. P1：完善输入校验与资源生命周期

### 11.1 不再静默忽略未知配置

当前 `LLMEngine` 过滤非 Config 字段。拼错配置时应抛出：

```text
Unknown config key: max_model_length
Did you mean: max_model_len?
```

### 11.2 校验请求

入口应验证：

- prompts 非空；
- token IDs 列表非空；
- prompts 与 SamplingParams 数量一致；
- `max_tokens > 0`；
- `prompt_len + max_tokens <= max_model_len`；
- token ID 在词表范围内；
- temperature 和未来 sampling 参数合法。

### 11.3 明确返回类型

修正 `generate()` 类型标注，最好定义：

```python
@dataclass
class RequestOutput:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
```

### 11.4 资源清理

建议：

- `LLM` 实现 context manager；
- `exit()` 幂等；
- 子进程启动失败时回收已启动进程；
- join 有超时和 terminate fallback；
- SharedMemory 创建失败时清理；
- 动态选择分布式端口；
- SharedMemory 名称加入 PID/UUID；
- 捕获子进程异常并传回 rank 0。

### 11.5 验收指标

- 错误输入产生明确异常；
- 异常初始化后无残留子进程和共享内存；
- 同一机器可运行多个实例；
- `exit()` 多次调用不崩溃；
- 类型检查与实际返回一致。

---

## 12. P2：优化 KV Cache 与前缀缓存

### 12.1 增加可观测性

先记录：

- 总 block/空闲 block/使用中 block；
- Prefix Cache query/hit blocks；
- Prefix Cache hit tokens；
- block 平均引用计数；
- block 生命周期；
- 从释放到再次复用的空闲时间；
- eviction 和 recompute 成本。

### 12.2 优化 Hash 计算

当前每次 hash 都把 token list 转为 NumPy 数组。可考虑：

- Sequence 追加完整 block 时增量生成 bytes/hash；
- 保存每个逻辑 block 的 token tuple 与链式 hash；
- 避免 allocate 时再次计算已知 hash；
- 将 hash metadata 与 Sequence 生命周期绑定；
- 对不可信多租户场景允许更强 hash 算法。

### 12.3 分离物理 Page 与逻辑匹配粒度

当前 FlashAttention paged KV 要求 page block size 为 256 的倍数，不能只删除 Config 断言。

长期可以设计：

```text
物理 KV page       256 tokens
逻辑 prefix unit    32/64 tokens
```

难点是命中物理 page 中间位置时，如何安全引用或复制尾部，以及如何与 `block_table` 和 slot mapping 协作。

另一条路线是抽象 Attention backend，使用支持更小 page 的 kernel。

### 12.4 淘汰策略

当前 free block deque 隐含近似 FIFO 行为。可比较：

- FIFO；
- LRU；
- LFU；
- 基于前缀深度；
- 基于引用/复用概率；
- tenant-aware；
- 保留系统提示词等高价值前缀。

淘汰策略不能只看 hit rate，还要衡量节省的 Prefill tokens。

### 12.5 Cache-aware Scheduling

当多个 waiting 请求都能运行时，可以优先：

- Prefix Cache 命中更多 token 的请求；
- 需要新 block 更少的请求；
- 能快速完成 Prefill 并释放资源的请求。

但必须结合 aging，避免没有公共前缀的请求饥饿。

### 12.6 KV Cache 量化

可研究 FP8 KV Cache：

- per-tensor scale；
- per-head scale；
- per-block scale；
- 动态或离线 scale；
- K/V 分别 scale。

它可以增加 KV 容量，但需要：

- Attention kernel 支持；
- 写入时量化；
- 读取时正确缩放；
- 长上下文精度评估；
- 不同 GPU 架构兼容测试。

### 12.7 CPU Offload 的定位

CPU KV offload 可以避免部分重算，但会引入 PCIe 传输和复杂状态管理。对于保持代码简洁的项目，recompute 往往更合理。

只有在测量表明重算代价明显高于传输代价时，才建议加入 offload，并应作为可选后端。

---

## 13. P2：优化 CUDA Graph 与编译策略

### 13.1 优化 Graph Bucket

当前 bucket 固定。可增加统计：

- 每个真实 batch size 出现次数；
- 映射到 bucket 后的 padding 数；
- padding FLOPs 比例；
- 每个 Graph 捕获时间和内存。

再比较：

```text
现有：1,2,4,8,16,32,48,...
2 的幂：1,2,4,8,16,32,64,...
流量驱动：只捕获最常见 batch size
按需捕获：首次遇到时生成
```

### 13.2 Graph 捕获范围

当前主要捕获 Transformer model。可以评估把以下部分纳入：

- LM Head；
- greedy sampler；
- temperature sampler；
- TP collective。

纳入随机采样时必须验证 RNG 在 replay 中正确推进。

### 13.3 Graph 内存与 KV Cache 规划

应显式记录：

```text
模型权重
eager 峰值工作区
CUDA Graph private pool
长期输入/output buffers
KV Cache
安全余量
```

让 KV block 数基于完整启动后的真实可用空间，而不是只依赖单次 warmup 的近似值。

### 13.4 `torch.compile` 策略

比较：

- 默认 compile；
- `reduce-overhead`；
- `max-autotune`；
- 手工 CUDA Graph；
- 不同 shape padding。

必须记录首次编译时间、缓存复用和稳态性能，避免只报告 warmup 后的最佳数字。

---

## 14. P2：优化张量并行与进程通信

### 14.1 避免 Gather 完整 Logits

Greedy 可以这样实现：

1. 每个 rank 计算本地最大 logit 和本地 token ID；
2. rank 间只比较候选 `(score, global_token_id)`；
3. 选出全局最大 token。

随机采样可研究分布式 Gumbel-max：

1. 每个 rank 为自己的词表分片产生独立随机 score；
2. 每个 rank 只保留局部最大候选；
3. 跨 rank 比较少量候选；
4. 不再 gather `[batch, vocab]`。

这要求严格验证其概率分布与单卡一致。

### 14.2 通信与计算重叠

Row Parallel 层当前同步 `all_reduce`。可以研究：

- async collective；
- 独立通信 stream；
- reduce-scatter + all-gather；
- sequence parallel；
- grouped collective。

由于 Transformer 层有严格数据依赖，重叠空间有限，应以 Nsight 时间线为依据。

### 14.3 优化控制面 IPC

当前控制消息使用：

- Python pickle；
- 固定 1 MiB SharedMemory；
- Event；
- 固定 SharedMemory 名称。

可优化为：

- 结构化共享内存 ring buffer；
- 仅传递紧凑数值字段；
- 长期共享的 Sequence metadata slots；
- 请求增量更新而不是每步 pickle 整个对象；
- 消息长度边界检查；
- 唯一实例 ID；
- 子进程错误响应通道。

### 14.4 TP 的适用范围

对于 0.6B 小模型，多卡通信可能比单卡更慢。Benchmark 应明确报告：

- 模型是否单卡可容纳；
- TP 增加是为容量还是为吞吐；
- 每层通信占比；
- TP=1/2/4 的 scaling efficiency。

---

## 15. P2：分片加载模型权重

### 15.1 当前问题

每个 rank 调用 `get_tensor()` 得到完整权重，再在 `weight_loader` 中切片。多卡会重复加载完整 Tensor。

### 15.2 推荐方案

使用 Safetensors `get_slice()`：

```text
Column Parallel：只读取本 rank 的输出维切片
Row Parallel：只读取本 rank 的输入维切片
Vocab Parallel：只读取本 rank 的词表切片
QKV Packed：分别计算 q/k/v 在文件中的 rank 切片
Gate-Up Packed：分别读取 gate/up 的 rank 切片并写入目标区域
Replicated：仍读取完整 Tensor
```

### 15.3 需要重构的接口

当前 loader 把完整 Tensor 传给参数：

```python
weight_loader(param, loaded_weight, shard_id)
```

可以改为让参数提供切片计划：

```python
slice_spec = param.get_weight_slice(weight_shape, shard_id)
loaded_slice = safe_tensor_slice[slice_spec]
param.load_weight_slice(loaded_slice, shard_id)
```

### 15.4 验收指标

- 权重加载后与旧实现逐参数一致；
- TP=2/4 的 CPU 峰值内存下降；
- checkpoint 读取字节数下降；
- 启动时间不退化；
- 分片 checkpoint 和单文件 checkpoint 均正常。

---

## 16. P2：量化与 Kernel 优化

### 16.1 权重量化

候选方向：

- BF16/FP16 基线；
- FP8；
- INT8 weight-only；
- INT4/AWQ/GPTQ。

需要解决：

- checkpoint 格式；
- quantized linear kernel；
- packed QKV/Gate-Up；
- TP 切片；
- 不同 GPU 架构；
- 精度评估。

量化不应只报告模型权重变小，还要报告 Prefill/Decode 吞吐、KV 容量和输出质量。

### 16.2 Kernel 融合

可评估：

- QKV projection + reshape；
- Q/K Norm + RoPE；
- RoPE + KV Cache write；
- residual add + RMSNorm；
- Gate-Up + SiLU + multiply；
- LM Head + sampling；
- Decode metadata prepare。

项目已经用 `torch.compile` 处理部分算子，应先查看生成 kernel，再决定是否手写 Triton。

### 16.3 Attention Backend 抽象

建立小型接口：

```python
class AttentionBackend:
    def prefill(...): ...
    def decode(...): ...
    def supported_block_sizes(...): ...
```

这样可以对比 FlashAttention、FlashInfer 或自定义 Triton kernel，而不把 backend 判断散落到模型代码中。

但抽象应保持精简，避免为了尚不存在的后端提前设计复杂框架。

---

## 17. P3：投机解码与高级推理能力

### 17.1 投机解码

基本流程：

```text
小 draft 模型一次提出多个 token
-> 主模型一次验证多个 token
-> 接受匹配前缀
-> 从第一个拒绝位置重新采样
```

潜在收益来自减少主模型 Decode forward 次数，但收益依赖：

- draft 模型速度；
- 接受率；
- batch size；
- 主模型大小；
- 额外 KV Cache；
- 验证 kernel。

它会影响 Scheduler、Sequence、KV block rollback、采样正确性和 CUDA Graph，因此应在统一 token 调度与测试系统稳定后实现。

### 17.2 Multi-token Prediction

如果模型原生带 MTP head，可一次产生多个候选 token。它比独立 draft 模型减少额外权重，但同样需要验证、接受与 KV 提交机制。

### 17.3 Sliding Window

对支持局部 Attention 的模型，只保留最近窗口 KV 可以显著降低长上下文缓存。需要：

- 模型配置识别；
- KV block 回收；
- Attention window 参数；
- Prefix Cache 语义；
- 位置编码兼容。

### 17.4 更多模型

模型扩展建议在执行层接口稳定后进行。优先选择结构接近 Qwen3 的模型，并把：

- model registry；
- packed weight mapping；
- Attention backend；
- position encoding；
- quantization config

控制在小型、清晰的接口内。

---

## 18. P3：从离线引擎走向在线服务

在线服务并不只是包一层 HTTP。需要增加：

- 异步请求队列；
- 动态加入请求；
- 流式增量 token；
- 请求取消；
- 超时和 deadline；
- backpressure；
- admission control；
- stop strings；
- disconnect 清理；
- 健康检查；
- 指标接口；
- 优雅关闭；
- 多租户隔离。

### 18.1 流式输出

当前 `step()` 只收集完成请求。可以输出每轮增量：

```python
TokenUpdate(
    seq_id=...,
    token_id=...,
    text_delta=...,
    finished=...,
    finish_reason=...,
)
```

增量 decode 要处理 tokenizer 跨 token 字节边界，不能简单对每个 token 单独 `decode` 后拼接。

### 18.2 请求取消

取消必须：

- 从 waiting/running 中删除请求；
- 正确减少 block ref_count；
- 不影响共享 Prefix Cache；
- 处理已经发给 GPU、尚未返回的 step；
- 清理流式消费者。

### 18.3 Admission Control

根据以下因素决定接收、等待或拒绝：

- prompt 长度；
- max_tokens；
- 当前 KV 空闲容量；
- deadline；
- 租户配额；
- 预估 Prefill 成本。

---

## 19. Benchmark 设计

### 19.1 必须固定的环境信息

每份结果记录：

```text
Git commit
GPU 型号与数量
GPU driver
CUDA
PyTorch
Triton
FlashAttention
Python
模型与 dtype
tensor_parallel_size
enforce_eager
max_num_batched_tokens
max_num_seqs
max_model_len
gpu_memory_utilization
kvcache_block_size
```

### 19.2 Workload 矩阵

| 类型 | Input | Output | 并发 | 关注指标 |
|---|---:|---:|---:|---|
| 单请求 Decode | 32 | 512 | 1 | TPOT、CPU gap |
| 高并发 Decode | 32 | 512 | 128 | output tok/s |
| Prefill | 2048 | 1 | 1/32 | prompt tok/s、TTFT |
| Chunked Prefill | 16384 | 16 | 1/8 | chunk 开销、公平性 |
| 混合长度 | 分布采样 | 分布采样 | 128 | P95/P99 |
| Prefix Cache | 共享 256+ | 64 | 32 | hit tokens、TTFT |
| 内存压力 | 长上下文 | 长输出 | 尽可能多 | preemption、重算 |
| 多卡 | 多组长度 | 多组长度 | 多组 | scaling efficiency |

### 19.3 预热和测量

- 初始化时间单独报告；
- compile/CUDA Graph capture 单独报告；
- 稳态 benchmark 前预热；
- 至少重复多轮；
- 报告均值、中位数、P95/P99 和标准差；
- 测量期间确认没有其他 GPU 负载；
- 随机输入和真实对话输入分别测试。

### 19.4 性能回归阈值

建议 CI 或固定 GPU 机器设置：

```text
正确性：必须通过
吞吐回归：超过预设比例则报警
TTFT/TPOT 回归：超过预设比例则报警
显存峰值：超过预算则报警
初始化时间：明显增长则报警
```

阈值应基于机器稳定性测量确定，不要凭空指定。

---

## 20. 测试策略

### 20.1 CPU 单元测试

无需 GPU 即可覆盖：

- Sequence 属性和状态；
- BlockManager 分配/释放；
- Prefix Cache 命中；
- hash collision 防护；
- ref_count；
- Scheduler token budget；
- Chunked Prefill；
- 抢占顺序；
- 公平性；
- 输入校验。

当前包顶层 import 会拉入 GPU 依赖，可考虑让管理层模块可独立导入，方便轻量 CI。

### 20.2 GPU 正确性测试

- 单层算子与 PyTorch reference；
- Qwen3 小配置随机权重；
- checkpoint 加载；
- Prefill logits；
- 多步 Decode；
- FlashAttention 与参考 Attention；
- eager/CUDA Graph；
- Prefix Cache；
- Chunked Prefill；
- TP。

### 20.3 属性测试

随机生成：

- 序列长度；
- block 分配顺序；
- cache hit/miss；
- append/deallocate；
- preemption；

并检查不变量：

```text
ref_count 永不小于 0
used 与 free 不相交
每个 block 恰好属于正确集合
Sequence block_table 长度足够
释放后引用计数正确
缓存命中 token 与原 token 完全一致
```

### 20.4 组合测试

最容易出错的是功能组合：

```text
Prefix Cache + Chunked Prefill
Prefix Cache + Preemption
CUDA Graph + TP
Sampling + TP
取消 + 共享 Prefix Cache
KV 量化 + Prefix Cache
Spec Decode + Preemption
```

每增加高级功能，都要明确它与已有功能的组合支持矩阵。

---

## 21. 推荐实施里程碑

### M0：基线与护栏

交付：

- CPU 单元测试框架；
- deterministic greedy；
- TTFT/TPOT/ITL/吞吐指标；
- Prefix Cache 与 preemption 指标；
- 固定 workload benchmark；
- Profiler/NVTX 标记。

完成后才能可信比较后续优化。

### M1：低风险 Quick Wins

交付：

- 跳过中间 Prefill chunk 的 LM Head/Sampler；
- 修复 API 类型与输入校验；
- Sampler 去除 Softmax；
- 配置错误不再静默忽略；
- 动态端口/共享内存名称；
- 幂等资源清理。

### M2：调度与 Host Overhead

交付：

- Decode 优先/平衡调度策略；
- 统一 token budget 数据模型；
- reusable pinned/GPU buffers；
- persistent Decode batch；
- Graph bucket 统计和调整；
- 公平性与延迟 benchmark。

### M3：内存与多卡

交付：

- Safetensors 分片读取；
- 分布式 greedy/采样；
- Prefix Cache hash/eviction 优化；
- TP 控制面 IPC 优化；
- KV Cache 内存规划改进。

### M4：实验性高级能力

交付候选：

- FP8 KV Cache；
- 权重量化；
- Speculative Decoding；
- Attention backend 抽象；
- 在线流式服务。

每项应独立开关，避免影响 Nano-vLLM 的默认教学路径。

---

## 22. 推荐的 PR 拆分方式

不要把调度、采样、KV Cache 和 API Server 放进一个巨大 PR。推荐：

```text
PR 1  Benchmark metrics + tests
PR 2  Greedy sampling + deterministic correctness harness
PR 3  Skip sampling for incomplete prefill chunks
PR 4  Remove sampler softmax
PR 5  Input validation and lifecycle cleanup
PR 6  Reusable decode metadata buffers
PR 7  Decode-priority scheduler policy
PR 8  Unified scheduler token accounting
PR 9  Safetensors rank-local slicing
PR 10 Distributed sampling without full-logit gather
```

每个性能 PR 应包含：

1. 修改前 benchmark；
2. 修改后 benchmark；
3. 正确性测试；
4. 显存变化；
5. 复杂度和限制说明；
6. 回滚或关闭开关。

---

## 23. 不建议过早进行的优化

### 23.1 一开始就支持大量模型

模型数量增加会快速扩大测试矩阵。在 Scheduler、KV Cache、加载器接口尚未稳定前，过早扩展模型会让重构成本倍增。

### 23.2 没有 Profile 就手写大量 Triton

手写 kernel 维护成本高。先确认算子真的是瓶颈，并比较 `torch.compile` 生成结果。

### 23.3 只追求 README 中一个 tokens/s 数字

单一合成 benchmark 可能隐藏 TTFT、P99、显存和初始化退化。所有优化至少报告吞吐和延迟两个维度。

### 23.4 直接把 KV block size 改小

当前 FlashAttention paged KV 对 page size 有约束。直接删除断言可能导致运行时报错或进入未经验证的 kernel 路径。

### 23.5 先做完整 OpenAI API Server

如果核心引擎还不能异步加入、取消和流式返回请求，HTTP 层只会掩盖真正需要解决的状态管理问题。

### 23.6 同时引入量化、Spec Decode 和新 Scheduler

它们都会改变 token、KV 和调度语义，组合调试非常困难。应该逐项建立正确性基线。

---

## 24. 最推荐优先落地的三个改动

### 第一：Benchmark 与测试基础设施

原因：没有可复现的 TTFT、TPOT、吞吐、缓存命中和重算指标，任何优化结论都不可靠。

### 第二：跳过中间 Prefill 采样，并优化 Sampler

原因：这是当前源码中清楚可见的重复计算，改动相对局部，容易验证，也能为 greedy 正确性测试打基础。

### 第三：Reusable Decode Buffers + 调度策略

原因：Decode 每轮只处理少量 token，Host 元数据开销和 Prefill 阻塞更容易成为用户可感知的延迟来源。这两项最能体现推理引擎区别于普通模型 forward 的价值。

推荐实施顺序：

```text
指标/测试
-> greedy
-> 跳过中间 Prefill 采样
-> Sampler 去 Softmax
-> reusable buffers
-> Decode-priority policy
-> unified token-budget scheduler
```

---

## 25. 参考资料

- [vLLM Scheduler 配置](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)
- [vLLM V1 Scheduler](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)
- [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [vLLM Cache 配置](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
- [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
- [PyTorch torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [Safetensors 文档](https://huggingface.co/docs/safetensors/index)
- [FlashAttention README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)

---

## 结语

Nano-vLLM 未来最值得优化的，不只是某个矩阵乘法或某个 kernel，而是一次 token 生成周围的整个系统成本：

```text
等待多久被调度
-> CPU 准备多少元数据
-> 搬运多少字节
-> GPU 做多少有效计算
-> KV Cache 是否被复用
-> 多卡通信多少数据
-> 结果是否及时返回用户
```

坚持“先测量、再优化；先正确、再复杂；保留可读的基线路径”，才能在提高性能的同时继续保持 Nano-vLLM 的核心价值。
