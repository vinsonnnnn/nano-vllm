# 文件职责：实现当前引擎实际执行的 Qwen3 网络，不调用 Transformers.generate。
# Runner 创建 Qwen3ForCausalLM；主干调用 Embedding/Decoder/RMSNorm，输出头另行计算。
# 输入 GPU int64 input_ids/positions [T]，输出主干 hidden_states [T,H]。
# 关键概念：GQA 的 Q/KV 头数差异、每卡投影分片、残差流、RoPE、输出权重共享。
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """QKV 投影 → 可选 Q/K Norm → RoPE → Attention → 输出投影。

    每卡处理 Nq=num_heads/P 个 Q 头与 Nkv=num_kv_heads/P 个 KV 头；头数需整除 P。
    不实现 KV 头跨 rank 复制模式；缓存写入和历史读取由 layers.Attention 完成。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        """构造每卡 Attention 分片；hidden_size 为 H，head_dim 为 D，max_position 为位置上限。

        scale=D**(-0.5) 缩放 QK 分数；本实现仅在无 qkv_bias 时创建 Q/K RMSNorm。
        rope_scaling 在这里只读取 rope_theta，不代表支持所有 RoPE scaling 算法。
        """
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 打包投影输出的本卡宽度：Q=Nq*D，K/V=Nkv*D，不能假定三段等宽。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """将 GPU [T,H] 输入转为 GPU [T,H] 输出；positions [T] 是逻辑位置。

        中间 Q [T,Nq,D]，K/V [T,Nkv,D]；更新本层 KV，输出投影可能触发 all_reduce。
        """
        # 步骤1：一次线性投影得到本卡打包 Q/K/V，按真实宽度 split 而非均分。
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 步骤2：每个头沿 D 做 Q/K Norm（满足当前配置条件时），然后按 position 旋转。
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        # 步骤3：Attention 返回本卡头输出；flatten 拼回 Nq*D 后做行并行输出投影。
        # o 在 Prefill 通常 [T,Nq,D]；Decode 接口可带长度为 1 的 query 轴。
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """门控前馈网络：down(SiLU(gate(x)) * up(x))，不是 Attention 或采样。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """创建打包 Gate/Up 列并行投影和 Down 行并行投影；intermediate_size 是全局 I。"""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """GPU [T,H] → [T,2I/P] → [T,I/P] → [T,H]；Down 跨 rank 求和合并。"""
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """一个 Transformer 层：Norm/Attention 与 Norm/MLP，保留独立 residual 传递。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """由 Qwen3Config 创建 Attention、MLP 及两个 H 维 RMSNorm；不分配独立 KV 池。"""
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """输入 position [T]、分支 hidden_states [T,H] 与 residual，返回两个 [T,H]。

        首层 residual=None；随后 Norm 合并上个子层输出与残差并保留更新后的残差。
        本层返回 MLP 分支输出，最后一次残差合并延迟到下一层/模型末尾 Norm。
        """
        # 步骤1：首次保存初始残差；后续以 add+RMSNorm 合并上轮分支，避免另写加法。
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 步骤2：Attention 分支 → 合并残差并归一化 → MLP 分支。
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """网络主干：Embedding → 多个 Decoder Layer → 最终 Norm；不产生词表 logits。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """按配置创建词表分片、层列表及最终归一化；参数数值之后由 loader 读入。"""
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """将 GPU token IDs/positions [T] 转成最终 hidden_states [T,H]，供 LM Head 打分。"""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """主干与 LM Head 的容器；forward 返回特征，compute_logits 才返回词表分数。"""
    # 权重文件中的分离投影名 → 本地打包参数名/片段 ID，供 loader 按片段装入。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        """创建主干和词表输出头；tie_word_embeddings=True 时让二者共享权重存储。"""
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Runner 调用的主干入口；不在这里采样，CUDA Graph 捕获的主要就是这一路径。"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """由最终 GPU 特征计算每请求 logits [B,V]；多卡只有 rank 0 拼出完整词表。"""
        return self.lm_head(hidden_states)
