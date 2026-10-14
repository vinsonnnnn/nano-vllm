# 文件职责：构造随机 token 输入，测量离线生成的总输出吞吐，而非在线服务延迟。
# main() 调用 LLM.generate；不测 P99、请求到达排队时间或独立 TTFT/ITL。
# 关键概念：预热与计时区间、输出 token 数、CUDA Graph；对照实验需要一致环境。
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    """在 CUDA 上生成 256 条随机请求，打印输出 tokens/s；随机输入不评估文本质量。"""
    # 步骤1：固定 Python 随机种子，使输入/输出长度可复现；未固定 torch 采样 RNG。
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 步骤2：直接提供 CPU token ID 列表，绕过文字编码。
    # ignore_eos=True 避免提前因 EOS 结束；上限来自每条请求的 max_tokens。
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 步骤3：先预热一次再计时；构造/加载模型不在计时区间内。
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    # 步骤4：用配置的输出总数除以整次 generate 墙钟时间。
    # 分母仍包括调度、CPU/GPU 交互和结果解码，不是仅 GPU kernel 的时间。
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
