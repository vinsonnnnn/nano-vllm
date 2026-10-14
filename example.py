# 文件职责：演示离线文本生成，入口是 main()；不实现 HTTP 服务或流式接口。
# 调用链：聊天模板 → LLM.generate → 引擎/调度/GPU 执行 → 返回文字与 token IDs。
# 关键概念：聊天模板只组织文字；tokenizer 编码由 LLMEngine.add_request 完成。
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """加载本地 Qwen3，生成两条请求并打印结果；需要 CUDA 及已下载的模型权重。"""
    # 步骤1：读取配置/tokenizer 并初始化引擎；初始化会加载权重、预热和分配 KV 池。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # max_tokens 是新增 token 上限，不是 prompt 长度；不是保证得到完整句子。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 步骤2：增加角色分隔和 assistant 起始提示。tokenize=False 表示仍返回字符串。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 步骤3：同步等待所有请求完成；返回按输入顺序排列的字典列表。
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
