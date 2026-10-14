# 文件职责：读取本地 safetensors，将训练后的数值装入已经创建的模型参数。
# Runner 初始化调用 load_model；各层参数的 weight_loader 实现打包/TP 切片规则。
# CPU 读取完整源 Tensor，copy_ 写入参数设备（正常为 GPU）；本模块不执行 forward。
# 关键概念：权重文件名称与本地打包名称不同，分片 ID 不等于 TP rank。
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """无专用加载器时整份复制 CPU 权重到目标参数；不自行推断 TP 切分。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """遍历 path 下 *.safetensors 并写模型参数；无返回值，输入模型被原地填充。

    packed_modules_mapping 来自 Qwen3ForCausalLM；参数 weight_loader 来自计算层。
    当前没有独立的权重完整性清单检查；缺失/不兼容文件不一定能在此统一诊断。
    """
    # 步骤1：在 CPU 打开权重文件，遍历各参数名称；避免先在 GPU 放一整份源权重。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 步骤2：q/k/v 或 gate/up 名称映射到本地打包参数，交给分支加载器。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                # 步骤3：for...else 仅在没有 break（无打包名称命中）时进入。
                # 普通参数优先用专用 TP 加载器，否则使用完整复制。
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
