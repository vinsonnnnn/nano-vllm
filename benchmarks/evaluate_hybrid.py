"""Reproducible CUDA latency/memory, LongBench-v1 subset and Needle ablation.

python -m benchmarks.evaluate_hybrid --model PATH --output reports/qwen35
Downloads pinned public LongBench data/metrics into .work (not the repository).
Results are flushed per case so an interrupted run preserves completed evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
import zipfile

import torch

from nanovllm import HybridLLM, SamplingParams
from nanovllm.engine.hybrid_cache import CompressionConfig

MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"


def prepare_longbench(root):
    import httpx
    from huggingface_hub import hf_hub_download
    code = root / "longbench-code"
    code.mkdir(parents=True, exist_ok=True)
    for name in ["dataset2prompt.json", "dataset2maxlen.json", "metrics.py"]:
        path = code / name
        remote = "config/" + name if name.endswith("json") else name
        # Always verify the exact pinned reference, not a mutable local copy.
        url = f"https://raw.githubusercontent.com/THUDM/LongBench/{LONGBENCH_REVISION}/LongBench/{remote}"
        path.write_bytes(httpx.get(url, follow_redirects=True, timeout=60).raise_for_status().content)
    archive = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset",
                              revision=DATA_REVISION, local_dir=root / "longbench")
    spec = importlib.util.spec_from_file_location("longbench_metrics", code / "metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return zipfile.ZipFile(archive), json.loads((code/"dataset2prompt.json").read_text(encoding="utf-8")), json.loads(
        (code/"dataset2maxlen.json").read_text(encoding="utf-8")), metrics


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[round((len(values)-1)*p)]


def render_report(records, metadata, target):
    perf = [r for r in records if r["suite"] == "performance"]
    lines = ["# Qwen3.5 性能与质量实测报告", "", "本报告由实际运行日志生成；所有性能数值只适用于本次环境。",
             "", "## 环境", "", "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```", "",
             "## 性能消融", "", "串行单请求；每组预热后重复测量。TTFT 含编码、状态分配、预填充和压缩；TPOT=(末次输出−首次输出)/(输出数−1)。",
             "解码吞吐=(输出数−1)/(末次输出−首次输出)，排除首 Token；系统吞吐含预填充。压缩 gather 的瞬时双份存储包含在 CUDA 峰值中。",
             "", "| 输入 Token | 配置 | 次数 | TTFT p50 / p95 ms | TPOT p50 ms | 解码 tok/s | 最终 KV MiB | CUDA allocated 峰值 MiB |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    groups = sorted({(r["requested_length"], r["mode"]) for r in perf})
    for length, mode in groups:
        group = [r for r in perf if (r["requested_length"], r["mode"]) == (length, mode)]
        ms = [r["metrics"] for r in group]
        ttft = [m["ttft_ms"] for m in ms]
        decode_seconds = sum(sum(m["itl_ms"])/1000 for m in ms)
        rate = sum(m["output_tokens"]-1 for m in ms)/decode_seconds
        lines.append(f"| {length} | {mode} | {len(group)} | {percentile(ttft,.5):.2f} / {percentile(ttft,.95):.2f} | "
                     f"{percentile([m['tpot_ms'] for m in ms],.5):.2f} | {rate:.2f} | "
                     f"{statistics.mean(m['kv_final_bytes'] for m in ms)/2**20:.2f} | "
                     f"{max(r['cuda_peak_allocated'] for r in group)/2**20:.2f} |")
    lines += ["", "KV 是存活 K/V 张量的实际字节数；不等于 CUDA allocator reserved，也不包含 GDN、卷积、观测窗口和临时 gather。",
              "每组仅 3 次时 p95 接近样本最大值，不能据此判断尾延迟稳定性。TTFT 差异可能受 WDDM、GPU 温度和系统负载影响。",
              "原始 JSONL 同时记录上述状态字节数、CUDA allocated/reserved、每 Token 间隔、生成文本和输入哈希。", "",
              "## LongBench v1 子集", "", "使用锁定版本的官方 prompt、任务最大输出长度和任务评分函数；取每任务前 N 条，使用当前 tokenizer 在中间截断至长度上限。",
              "本次对官方 prompt 使用 Qwen chat template，关闭 thinking。中间截断可能移除答案依据。结果仅为截断的小样本诊断，不能作为完整 LongBench 成绩或无损证明。", "",
              "| 任务 | 配置 | 样本数 | 官方任务指标均分 / 100 |", "|---|---|---:|---:|"]
    quality = [r for r in records if r["suite"] == "longbench"]
    for task, mode in sorted({(r["task"], r["mode"]) for r in quality}):
        rows = [r for r in quality if (r["task"], r["mode"]) == (task, mode)]
        lines.append(f"| {task} | {mode} | {len(rows)} | {100*statistics.mean(r['score'] for r in rows):.2f} |")
    lines += ["", "## Needle", "", "确定性数字口令；在指定背景 Token 深度插入，按解码文本中的独立数字精确匹配；单一背景、每个点一次，不代表通用检索能力。", "",
              "| 背景 Token | 深度 | 配置 | 命中 |", "|---:|---:|---|---|"]
    for r in records:
        if r["suite"] == "needle":
            lines.append(f"| {r['requested_length']} | {r['depth']:.0%} | {r['mode']} | {r['score']} |")
    errors = [r for r in records if r.get("error")]
    lines += ["", f"记录数：{len(records)}；失败数：{len(errors)}。", "",
              "## 适用范围", "", "这是参考 kernel 的单设备 eager 后端；Windows 上 GDN 使用 PyTorch fallback。原 Qwen3 的原生分页/FlashAttention 路径保留。",
              "压缩只发生在 Full Attention 层，各层共享头选择但独立选位置；它不缩短 GDN 已消费的历史。",
              "未验证的能力不计入性能结论：多卡 TP、CUDA Graph、FP8 GEMM、真实 FP8 检查点、完整 LongBench、随机投机采样。",
              "MTP/回滚、图文输入及状态兼容性的额外证据见 compatibility.json 和项目改造说明。", ""]
    target.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="reports/qwen35")
    p.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--interval", type=int, default=32)
    p.add_argument("--quality-limit", type=int, default=4096)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"results.jsonl").exists() and not args.resume:
        raise FileExistsError("use a fresh output directory to preserve previous evidence")
    torch.manual_seed(20261104)
    metadata = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "working_tree": "implementation under test; see committed source manifest",
                "model_revision": MODEL_REVISION, "longbench_data_revision": DATA_REVISION,
                "longbench_code_revision": LONGBENCH_REVISION, "dtype": "bfloat16", "tp": 1,
                "graph": False, "backend": "HF reference GDN fallback + compact SDPA",
                "seed": 20261104, "args": vars(args)}
    import transformers
    metadata["transformers"] = transformers.__version__
    source_files = sorted(list(Path("nanovllm").rglob("*.py")) + list(Path("benchmarks").glob("*.py")) + [Path("pyproject.toml")])
    metadata["source_sha256"] = {str(f).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    metadata["model_config_sha256"] = hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest()
    metadata["chunk_size"] = max(args.lengths + [args.quality_limit]) + 4096
    metadata["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"], text=True).strip()
    records = []
    if args.resume:
        original = json.loads((output/"environment.json").read_text(encoding="utf-8"))
        for key in vars(args):
            if key not in {"resume"} and original["args"][key] != vars(args)[key]:
                raise ValueError(f"resume configuration changed: {key}")
        changes = {name: {"before": digest, "after": metadata["source_sha256"].get(name)}
                   for name, digest in original["source_sha256"].items()
                   if metadata["source_sha256"].get(name) != digest}
        if any(not name.startswith("benchmarks/") for name in changes):
            raise ValueError("runtime source changed; use a fresh benchmark output directory")
        if original["model_config_sha256"] != metadata["model_config_sha256"]:
            raise ValueError("model config changed since the original run")
        current_hashes = metadata["source_sha256"]
        metadata = original
        if changes:
            metadata.setdefault("resume_source_changes", []).append(changes)
            metadata["source_sha256"] = current_hashes
        metadata.setdefault("resumed_utc", []).append(datetime.now(timezone.utc).isoformat())
        records = [json.loads(line) for line in (output/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    (output/"environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    llm = HybridLLM(args.model, max_model_len=max(args.lengths + [args.quality_limit])+4096,
                    max_num_batched_tokens=metadata["chunk_size"])
    modes = {"baseline": CompressionConfig(), "compressed": CompressionConfig(
        budget=args.budget, window=args.window, interval=args.interval)}

    def run(suite, prompt, mode, max_tokens, **details):
        llm.compression = modes[mode]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = perf_counter()
        result = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=max_tokens,
                              ignore_eos=suite == "performance"))[0]
        torch.cuda.synchronize()
        elapsed = perf_counter()-start
        row = {"suite": suite, "mode": mode, **details, "text": result["text"],
               "metrics": result["metrics"], "elapsed_s": elapsed,
               "system_output_tokens_per_second": len(result["token_ids"])/elapsed,
               "cuda_before_allocated": before, "cuda_after_allocated": torch.cuda.memory_allocated(),
               "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
               "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
               "prompt_sha256": hashlib.sha256(json.dumps(prompt,ensure_ascii=False).encode()).hexdigest()}
        return row

    def save(row):
        records.append(row)
        with (output/"results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
        print(json.dumps({k:row[k] for k in ["suite", "mode", "requested_length", "task", "score"] if k in row}), flush=True)
        render_report(records, metadata, output/"PERFORMANCE_REPORT_ZH.md")

    # Warm both execution branches; warmup cases are excluded from measurements.
    for mode in modes:
        run("performance", [101]*600, mode, 8)
    background = llm.tokenizer.encode("A quiet library stores books about geography, astronomy, architecture and local history. "*3000)
    for length in args.lengths:
        prompt = background[:length]
        if len(prompt) != length:
            raise ValueError("background is shorter than requested performance length")
        # Alternate modes per repeat to limit ordering/power-temperature bias.
        for repeat in range(args.repeats):
            for mode in (list(modes) if repeat%2 == 0 else list(reversed(modes))):
                if any(r["suite"] == "performance" and r["requested_length"] == length and
                       r["repeat"] == repeat and r["mode"] == mode for r in records):
                    continue
                save(run("performance", prompt, mode, args.output_tokens,
                         requested_length=length, repeat=repeat))
    if not args.skip_quality:
        archive, templates, maxlens, metrics = prepare_longbench(Path(".work"))
        for task in ["hotpotqa", "qasper", "passage_retrieval_en"]:
            samples = archive.read(f"data/{task}.jsonl").decode().splitlines()[:args.samples]
            for sample_id, line in enumerate(samples):
                sample = json.loads(line)
                raw = templates[task].format(**sample)
                ids = llm.tokenizer.encode(raw)
                original_length = len(ids)
                if len(ids) > args.quality_limit:
                    half = args.quality_limit//2
                    ids = ids[:half]+ids[-(args.quality_limit-half):]
                raw = llm.tokenizer.decode(ids)
                prompt = llm.tokenizer.apply_chat_template([{"role":"user","content":raw}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                for mode in modes:
                    if any(r["suite"] == "longbench" and r["task"] == task and
                           r["sample_id"] == sample_id and r["mode"] == mode for r in records):
                        continue
                    row = run("longbench", prompt, mode, maxlens[task], task=task,
                        sample_id=sample_id, original_tokens=original_length,
                        truncated_tokens=len(ids), answers=sample["answers"])
                    metric = metrics.retrieval_score if task.startswith("passage") else metrics.qa_f1_score
                    row["score"] = max(metric(row["text"], answer) for answer in sample["answers"])
                    save(row)
        archive.close()
        import re
        for length in args.lengths:
            for depth in [.1, .5, .9]:
                secret = str(710000 + length + round(depth*100))
                needle = llm.tokenizer.encode(f"\nThe secret archive access code is {secret}.\n")
                split = int(length*depth)
                haystack = llm.tokenizer.decode(background[:split]+needle+background[split:length])
                prompt = llm.tokenizer.apply_chat_template([{"role":"user","content":
                    haystack+"\nWhat is the secret archive access code? Answer only with the digits."}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                for mode in modes:
                    if any(r["suite"] == "needle" and r["requested_length"] == length and
                           r["depth"] == depth and r["mode"] == mode for r in records):
                        continue
                    row = run("needle", prompt, mode, 16, requested_length=length, depth=depth, expected=secret)
                    row["score"] = int(secret in re.findall(r"\b\d+\b", row["text"]))
                    save(row)
    llm.exit()


if __name__ == "__main__":
    main()
