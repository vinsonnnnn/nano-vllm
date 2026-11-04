"""Dispatch models without importing the CUDA-only Qwen3 path eagerly."""

class LLM:
    def __new__(cls, model, **kwargs):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model)
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            from nanovllm.engine.hybrid_engine import HybridLLM
            return HybridLLM(model, **kwargs)
        if config.model_type != "qwen3":
            raise ValueError(f"unsupported model_type: {config.model_type}")
        from nanovllm.engine.llm_engine import LLMEngine
        return LLMEngine(model, **kwargs)
