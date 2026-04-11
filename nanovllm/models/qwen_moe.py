import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ColumnParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen2MoeAttention(nn.Module):

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
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        model_path = "~/huggingface/Qwen1.5-MoE-A2.7B"
        config = AutoConfig.from_pretrained(model_path)
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.top_k = config.num_experts_per_tok
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
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=None,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if hasattr(self, 'q_norm'):
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output



class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
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
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
class Qwen2MoeLocalMLP(nn.Module):
    """用于 EP 模式下的局部稀疏专家，不能包含跨卡通信算子"""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.intermediate_size=intermediate_size
        # 使用普通的 nn.Linear！
        # 注意：因为是 SwiGLU，gate 和 up 的输出维度合并为 intermediate_size * 2
        self.gate_up_proj = nn.Linear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()
        self.gate_up_proj.weight.weight_loader = self.gate_up_weight_loader

    def gate_up_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        """
        处理局部专家的 gate 和 up 权重合并。
        由于是纯 EP 并行，没有 TP 切片，只需要做简单的纵向拼接
        shard_id: 0 代表 gate_proj, 1 代表 up_proj
        """
        # loaded_weight 的 shape 是 [intermediate_size, hidden_size]
        # 根据 shard_id 计算应该塞进合并张量的上半区还是下半区
        start_idx = shard_id * self.intermediate_size
        end_idx = start_idx + self.intermediate_size
    
        param.data[start_idx:end_idx].copy_(loaded_weight)
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
class Qwen2MoeSparseMoeBlock(nn.Module):
    
    def __init__(
        self,
        config: any,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        
        # 获取并行参数
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        assert self.num_experts % self.tp_size == 0
        
        # 每个 Rank 负责的专家数量
        self.experts_per_rank = self.num_experts // self.tp_size
        self.start_idx = self.tp_rank * self.experts_per_rank
        self.end_idx = self.start_idx + self.experts_per_rank
        
        # Router: 负责计算专家评分 (ColumnParallel 会计算属于本地 Rank 的 Logits)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Experts: 每个 GPU 仅初始化自己负责范围内的专家
        self.experts = nn.ModuleList([
            Qwen2MoeLocalMLP(
                config.hidden_size, 
                config.moe_intermediate_size, 
                config.hidden_act
            ) for _ in range(self.experts_per_rank)
        ])
        
        # Shared Expert: Qwen特有的始终开启的专家 (在所有卡上复制一份)
        self.shared_expert = Qwen2MoeMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_act,
        )
        # 共享专家的门控开关
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def _route_local_experts(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        routed_hidden_states = torch.zeros_like(hidden_states)

        for local_idx, global_idx in enumerate(range(self.start_idx, self.end_idx)):
            token_mask = (selected_experts == global_idx).any(dim=-1)
            if not token_mask.any():
                continue

            expert_mask = selected_experts == global_idx
            expert_weights = (routing_weights * expert_mask).sum(dim=-1, keepdim=True)
            expert_output = self.experts[local_idx](hidden_states[token_mask])
            routed_hidden_states[token_mask] += expert_output * expert_weights[token_mask]

        return routed_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. 计算路由得分
        router_logits = self.gate(hidden_states)
 
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # 2. 选择全局 Top-K 专家
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # 3. 计算本 Rank 负责的专家输出，并汇总到全局结果
        final_hidden_states = self._route_local_experts(hidden_states, selected_experts, routing_weights)
        if self.tp_size > 1:
            dist.all_reduce(final_hidden_states, op=dist.ReduceOp.SUM)

        # 4. 加上 Shared Expert 结果
        shared_output = self.shared_expert(hidden_states)
        shared_weight = torch.sigmoid(self.shared_expert_gate(hidden_states))
        
        return final_hidden_states + (shared_output * shared_weight)




class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: any,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta = getattr(config, "rope_theta", 
                     config.__dict__.get("rope_theta", 1000000.0)),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen2MoeSparseMoeBlock(
            config=config
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2MoeModel(nn.Module):

    def __init__(
        self,
        config: any,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
 
    def __init__(
        self,
        config: any
    ) -> None:
        super().__init__()
        self.model = Qwen2MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
