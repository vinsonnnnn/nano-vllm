# 实测结论与验证边界

本次使用真实 Qwen3.5-0.8B BF16 检查点，在 RTX 4090 Laptop 16GB / Windows / PyTorch 2.10.0 CUDA 12.8 上测试。**主要收益是存活 KV 容量降低；16K 时有解码收益，同时存在任务质量代价。** 全量表格见 [性能报告](PERFORMANCE_REPORT_ZH.md)，逐案例数据见 [results.jsonl](results.jsonl)。

## 16K 性能结论

| 指标 | 不压缩 | 压缩 | 解释 |
|---|---:|---:|---|
| 首 Token 延迟 p50 | 4063.95 ms | 4120.36 ms | 压缩没有缩短此项，增加约 1.4% |
| 单 Token 生成延迟 TPOT p50 | 42.21 ms | 34.68 ms | 本负载降低约 17.8% |
| 解码吞吐 | 23.80 Token/s | 28.78 Token/s | 本负载提高约 20.9% |
| 末次输出前存活 KV | 192.74 MiB | 6.36 MiB | 减少约 96.7%，不含 GDN/conv/观测缓冲 |
| CUDA allocated 峰值 | 3421.35 MiB | 3421.97 MiB | 基本不变，仍由完整预填充与临时张量主导 |

压缩参数为每个 Full Attention 层预算 512、近期窗口 32、周期 32、pooling 核 5。同层共享头的保留位置，各层独立选择。生成 64 Token，三个重复样本，单请求串行；此表不是高并发服务容量结论。1K/4K 下解码吞吐未获益，不能把 16K 的提升泛化到所有长度。

性能输入使用提前构造的 Token ID，因此此表的 TTFT **不包括字符串 tokenizer 或 chat template 的时间**，包括请求准备、embedding、状态分配、预填充、压缩和输出同步。引擎接受原始字符串/图像时，其请求 TTFT 会覆盖对应的编码和视觉处理。模型加载与预热均排除在性能案例之外。

同一进程内两条路径交替运行，并在输出可见前同步 GPU。WDDM、桌面进程和温度没有严格隔离，每组只有三次，p95 接近样本最大值；目前没有置信区间或跨设备复现。96.7% 是 KV 张量节省，不是总显存减少 96.7%，也不是吞吐提高 96.7%。

## 质量结论

LongBench **v1** 取三个任务各前四条，总计 12 个不同样本，分别执行 baseline 和 compressed 两组。使用固定官方模板、任务最大输出长度和官方评分函数；当前 tokenizer 中间截断到 8192 Token 后再应用关闭 thinking 的 Qwen chat template。

| 任务及指标 | 不压缩 | 压缩 | 变化 |
|---|---:|---:|---:|
| HotpotQA / F1 | 33.33 | 33.33 | 0.00 |
| Qasper / F1 | 12.28 | 7.14 | -5.14 分 |
| Passage retrieval en / 官方 retrieval score | 25.00 | 25.00 | 0.00 |

Qasper 表明这组激进预算有质量损失。其他两个任务在四个样本上相同，不构成无损证据；中间截断也可能移除答案所在段落，低基线成绩限制了判断能力。未将三个异质任务合并成“完整 LongBench 平均分”。

Needle 使用 1K/4K/16K 背景、10%/50%/90% 三个插入深度，两组均 **9/9** 精确命中。口令与背景为确定性构造，插入后加聊天提示，所以实际输入略长于背景长度；每个点仅一次，没有多个背景、多个随机口令或更大上下文的统计保证。

## 正确性与混合状态兼容性

证据见 [compatibility.json](compatibility.json) 与 [单元测试 XML](unit-tests.xml)。

- 18 项 CPU 数值/生命周期/事务测试通过，包括 FP32 参考对齐、分块预填充、真实紧凑存储、随机槽位复用、压缩后交错请求、拒绝及异常恢复、合成 FP8 流式加载和分片、长预填充执行选择。
- 真实检查点的完整预填充与三步 teacher forcing，对官方 SDPA BF16 路径的最大 logits 绝对误差均为 **0**，argmax 相同。该结论限定于测试输入和所用后端。
- 真实 BF16 分块预填充最后一步 argmax 相同，最大 logits 绝对误差 **0.171875**；chunkwise 与 recurrent 浮点累加不保证逐位一致。
- KV 压缩把 73 个逻辑位置保留为每个 Full Attention 层 48 个物理位置，logical length 仍为 73；GDN recurrent 与 conv 张量逐位不变。
- 真实请求交错、取消复用及不压缩条件下抢占重算，均与隔离执行输出相同。压缩条件下的重算是新的近似历史，不承诺恢复原压缩快照。
- 真实 MTP 的 15 个权重张量已加载；草稿、target 验证、强制拒绝后 KV/GDN/conv 恢复通过。未压缩的 12 Token `generate_mtp` 输出与普通贪心相同；压缩组合执行了两次压缩并正常回收槽位。
- 图像 processor、视觉特征注入、Interleaved MRoPE 和压缩解码通过；64 个视觉 Token 全部保留，红色测试图输出 `red`。这是一张合成图片的集成检查，不是视觉质量基准。

## 尚未交付或验证的能力

新路径当前使用请求独占的动态紧凑 KV、单设备 eager 执行。**混合路径的原生分页池、共享前缀缓存、多卡 TP、CUDA Graph 尚未实现**，不能把原 Qwen3 路径中的这些能力写成 Qwen3.5 已支持。TP 和非 eager 配置会明确报错。

FP8 已实现明确 scale 契约下的 block checkpoint 反量化和合成检查点加载测试，但没有真实 FP8 检查点端到端证据，也没有 FP8 GEMM。MTP 是 serial target verification 原型，保存状态并重建 draft history，不宣称推理加速；随机投机采样没有实现。完整 LongBench 和更广泛组合还需独立验证。

初期诊断运行遇到 Windows 文本编码、聊天模板返回类型以及原生 GQA 超长预填充换页问题，均已修复；修改前的记录保留在本机 `.work`，未混入本报告的 60 个最终案例。最终执行的源码哈希与 [environment.json](environment.json) 一致。测试前 HEAD 用于标记基线，不能替代被测源码哈希。

`source_sha256` 记录本次实际文件字节，`source_git_blob_sha1` 记录提交中的 Git blob，后者便于跨 Windows CRLF / Linux LF 检出核对。
