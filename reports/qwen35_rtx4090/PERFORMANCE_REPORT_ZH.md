# Qwen3.5 性能与质量实测报告

请结合 [实测结论与验证边界](ANALYSIS_ZH.md) 阅读：其中解释 Token ID 输入的计时口径、Qasper 质量损失及未实现的高级能力。

本报告由实际运行日志生成；所有性能数值只适用于本次环境。

## 环境

```json
{
  "utc": "2026-11-04T11:44:47.396321+00:00",
  "platform": "Windows-11-10.0.22631-SP0",
  "python": "3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]",
  "torch": "2.10.0+cu128",
  "cuda": "12.8",
  "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
  "gpu_total_bytes": 17170956288,
  "git_base": "f91eeaf01816a6d2f899172b4ce74df35d59d490",
  "working_tree": "implementation under test; see committed source manifest",
  "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
  "longbench_data_revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
  "longbench_code_revision": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
  "dtype": "bfloat16",
  "tp": 1,
  "graph": false,
  "backend": "HF reference GDN fallback + compact SDPA",
  "seed": 20261104,
  "args": {
    "model": ".work/models/Qwen3.5-0.8B",
    "output": "reports/qwen35_rtx4090",
    "lengths": [
      1024,
      4096,
      16384
    ],
    "repeats": 3,
    "output_tokens": 64,
    "budget": 512,
    "window": 32,
    "interval": 32,
    "quality_limit": 8192,
    "samples": 4,
    "skip_quality": false,
    "resume": false
  },
  "transformers": "5.3.0",
  "source_sha256": {
    "benchmarks/check_compatibility.py": "1f43ac429b5d035893ed49a5a7acb5fa70455e6d460e655ff009b04cca9b4974",
    "benchmarks/evaluate_hybrid.py": "1e521f99c6ae40afc11738094079b519e341fc198ae65f65fda76fcb18bddbbc",
    "nanovllm/__init__.py": "9aa972168b484673f09eb76246b29f5107c93c724fd4a5ea3bbf8fec8c24a0f3",
    "nanovllm/config.py": "05a1a057a6c8b052eed0557578278241f4bf311c0e767853fa30d2c70f182015",
    "nanovllm/engine/block_manager.py": "db02280bd4c2d5a252aaa703aebbdedcedc6fbaa25a023eff0d0e1657294a367",
    "nanovllm/engine/hybrid_cache.py": "46d5d0c582ede66c793c2bd0113341b5571a6f81dacf5e2b79c4a0da433df61b",
    "nanovllm/engine/hybrid_engine.py": "d50f76af5300a6d5cf76ce1a03c742af49ffdee000fb077822fcda8a46dc1aa2",
    "nanovllm/engine/llm_engine.py": "4390e597b881cc96ca03ad7be7867c24b3bed6412017e92a1ad834284c68b7b3",
    "nanovllm/engine/model_runner.py": "276fc324278d685f0b1502c491ea5f1a128c8eaed38854f8f1e282e965f484c5",
    "nanovllm/engine/scheduler.py": "10a8aadd70609927434d5390af2852cb10ba0af7ccf0e18eac5ea2c2c253832a",
    "nanovllm/engine/sequence.py": "fa65eab1f03e59963f717c315ef8e8bc4c91f0d379f3aed03695633eeb8c571c",
    "nanovllm/engine/speculation.py": "4df66933c8ce509bad43c9a53eda5c636d752a54ef8495a0d78df3ee6ec0830b",
    "nanovllm/layers/activation.py": "f69ca00329b034bdd52f375da1400581ec98e5967326ab086f465d2b84516fe9",
    "nanovllm/layers/attention.py": "d9d8ef15c0254976c5569aa159c185150bf64fbad0129be3f9c07727e1e5640d",
    "nanovllm/layers/embed_head.py": "5983d52054e0ec03ebe4214ce3fbf28f6857ffead34d98697edaf66e1e4e77eb",
    "nanovllm/layers/layernorm.py": "1a21260180c29f223bdbd4d7529b8406cc4c7bf6c3a134b62c4f57ea694efcd3",
    "nanovllm/layers/linear.py": "d54f4cb3ce574fdc710ff31cdb04d76e20dbf9f4ac7457e2ca7333165ee0f243",
    "nanovllm/layers/rotary_embedding.py": "a5f16c9946a4a4a5d54327b36f38f763d562e028bc272b67c347c8e0d2167592",
    "nanovllm/layers/sampler.py": "eca02bad5c5602955ca45e3539002bf3a3766377f8d5c858cb0adf8cfd7a0cf7",
    "nanovllm/llm.py": "0bd63a9d8badde873275b406036246e5bc6eb3afefac2f989b2f1365347bc0f3",
    "nanovllm/models/qwen3.py": "5c0a3680f346ce2fe115dd636ab45bffaca32555460fbf552c22c293d0512cf8",
    "nanovllm/models/qwen3_5.py": "bbee0432e995bf6eee4a03d740326269968c075153792ac7459970f03c5b24f0",
    "nanovllm/models/qwen3_5_mtp.py": "23a6a7a1beeb661e85be5792c98e12d9f97769571edebc7b7fce685388665dbf",
    "nanovllm/sampling_params.py": "2cf6acf0ddb9843d21fe462bf5a5e5d82d28e848ddd58969b5d812c622bfd883",
    "nanovllm/utils/context.py": "ff00cf7a9442feb9055b4e63cca81b4b504d1c0294e7f4e0f2260ec86d8a620f",
    "nanovllm/utils/fp8.py": "a29138e0c187d2e4d69d4c656bf18cedda4d16c54e0d25b1753890e19d63d5c5",
    "nanovllm/utils/loader.py": "e47cafcc6bae67c03fefbbafd10d680dfb3e5c614cc8676cb2e43f48411f4fcb",
    "pyproject.toml": "9dc8837104a2f0ab8fcd80aa915f8fe30fc1dca38bf622b3091aeb9d58703124"
  },
  "model_config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
  "chunk_size": 20480,
  "nvidia_smi": "NVIDIA GeForce RTX 4090 Laptop GPU, 610.47, 16376 MiB, [N/A]"
}
```

## 性能消融

串行单请求；每组预热后重复测量。TTFT 含编码、状态分配、预填充和压缩；TPOT=(末次输出−首次输出)/(输出数−1)。
解码吞吐=(输出数−1)/(末次输出−首次输出)，排除首 Token；系统吞吐含预填充。压缩 gather 的瞬时双份存储包含在 CUDA 峰值中。

| 输入 Token | 配置 | 次数 | TTFT p50 / p95 ms | TPOT p50 ms | 解码 tok/s | 最终 KV MiB | CUDA allocated 峰值 MiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1024 | baseline | 3 | 538.76 / 558.58 | 34.73 | 28.78 | 12.74 | 1826.39 |
| 1024 | compressed | 3 | 443.50 / 495.24 | 35.55 | 28.27 | 6.36 | 1827.14 |
| 4096 | baseline | 3 | 1211.27 / 1227.90 | 34.97 | 28.66 | 48.74 | 3166.25 |
| 4096 | compressed | 3 | 1240.96 / 1290.51 | 35.14 | 28.35 | 6.36 | 3167.00 |
| 16384 | baseline | 3 | 4063.95 / 4146.66 | 42.21 | 23.80 | 192.74 | 3421.35 |
| 16384 | compressed | 3 | 4120.36 / 4163.82 | 34.68 | 28.78 | 6.36 | 3421.97 |

KV 是存活 K/V 张量的实际字节数；不等于 CUDA allocator reserved，也不包含 GDN、卷积、观测窗口和临时 gather。
每组仅 3 次时 p95 接近样本最大值，不能据此判断尾延迟稳定性。TTFT 差异可能受 WDDM、GPU 温度和系统负载影响。
原始 JSONL 同时记录上述状态字节数、CUDA allocated/reserved、每 Token 间隔、生成文本和输入哈希。

## LongBench v1 子集

使用锁定版本的官方 prompt、任务最大输出长度和任务评分函数；取每任务前 N 条，使用当前 tokenizer 在中间截断至长度上限。
本次对官方 prompt 使用 Qwen chat template，关闭 thinking。中间截断可能移除答案依据。结果仅为截断的小样本诊断，不能作为完整 LongBench 成绩或无损证明。

| 任务 | 配置 | 样本数 | 官方任务指标均分 / 100 |
|---|---|---:|---:|
| hotpotqa | baseline | 4 | 33.33 |
| hotpotqa | compressed | 4 | 33.33 |
| passage_retrieval_en | baseline | 4 | 25.00 |
| passage_retrieval_en | compressed | 4 | 25.00 |
| qasper | baseline | 4 | 12.28 |
| qasper | compressed | 4 | 7.14 |

## Needle

确定性数字口令；在指定背景 Token 深度插入，按解码文本中的独立数字精确匹配；单一背景、每个点一次，不代表通用检索能力。

| 背景 Token | 深度 | 配置 | 命中 |
|---:|---:|---|---|
| 1024 | 10% | baseline | 1 |
| 1024 | 10% | compressed | 1 |
| 1024 | 50% | baseline | 1 |
| 1024 | 50% | compressed | 1 |
| 1024 | 90% | baseline | 1 |
| 1024 | 90% | compressed | 1 |
| 4096 | 10% | baseline | 1 |
| 4096 | 10% | compressed | 1 |
| 4096 | 50% | baseline | 1 |
| 4096 | 50% | compressed | 1 |
| 4096 | 90% | baseline | 1 |
| 4096 | 90% | compressed | 1 |
| 16384 | 10% | baseline | 1 |
| 16384 | 10% | compressed | 1 |
| 16384 | 50% | baseline | 1 |
| 16384 | 50% | compressed | 1 |
| 16384 | 90% | baseline | 1 |
| 16384 | 90% | compressed | 1 |

记录数：60；失败数：0。

## 适用范围

这是参考 kernel 的单设备 eager 后端；Windows 上 GDN 使用 PyTorch fallback。原 Qwen3 的原生分页/FlashAttention 路径保留。
压缩只发生在 Full Attention 层，各层共享头选择但独立选位置；它不缩短 GDN 已消费的历史。
未验证的能力不计入性能结论：多卡 TP、CUDA Graph、FP8 GEMM、真实 FP8 检查点、完整 LongBench、随机投机采样。
MTP/回滚、图文输入及状态兼容性的额外证据见 compatibility.json 和项目改造说明。
