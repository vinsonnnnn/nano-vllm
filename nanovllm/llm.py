# 文件职责：提供用户侧类名 LLM；实际初始化、generate、step 都继承 LLMEngine。
# example.py/bench.py 创建 LLM，本文件没有额外包装或独立模型实现。
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """公开推理入口；接口与行为直接继承 engine/llm_engine.py 中的 LLMEngine。"""
    pass
