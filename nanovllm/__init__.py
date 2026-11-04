# 包的公开入口：让用户通过 from nanovllm import LLM, SamplingParams 使用引擎。
# 这里只导出名称；实现分别在 llm.py、sampling_params.py，不执行生成流程。
from nanovllm.sampling_params import SamplingParams


def __getattr__(name):
    # CPU tooling and the portable hybrid backend do not require Triton/NCCL.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    if name == "HybridLLM":
        from nanovllm.engine.hybrid_engine import HybridLLM
        return HybridLLM
    raise AttributeError(name)
