import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyinfra.engine.cache_manager import KVCacheManager
from tinyinfra.kernels.fused_mlp import fused_mlp_forward
from tinyinfra.kernels.Paged_Attn import launch_paged_attention
from tinyinfra.kernels.RMSNorm import fused_add_rmsnorm, rmsnorm
from tinyinfra.kernels.Rope import rope_inplace
from tinyinfra.models.config import QwenConfig


class RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> torch.Tensor:

        super().__init__()
        # Initialize the weight parameter for the RMS normalization layer
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Store the epsilon value for numerical stability
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rmsnorm(hidden_states, self.weight, self.eps)

class Qwen2Attention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_id: torch.tensor,
        layer_idx: int,
        batch_indices: torch.Tensor,
        kv_cache: "KVCacheManager",
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:

        # Extract the batch size, sequence length from the input hidden states
        batch_size, q_length, _ = hidden_states.size()

        # Project the hidden states to query, key and value matrices.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Transform (batch_size, num_heads, seq_length, head_dim)
        query_states = query_states.view(batch_size, q_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_length, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_length, self.num_kv_heads, self.head_dim)

        # flatten to 3D/1D 
        query_3d = query_states.view(batch_size * q_length, self.num_heads, self.head_dim)
        key_3d = key_states.view(batch_size * q_length, self.num_kv_heads, self.head_dim)
        pos_id_flat = pos_id.reshape(-1).to(torch.int32)
        
        # Hardware accelerated in place rope
        rope_inplace(query_3d, key_3d, cos_cache, sin_cache, pos_id_flat)

        # Handover the cached key and value states to the KVCacheManager for updating and fetching
        block_tables, context_length = kv_cache.allocate_and_insert(
            layer_idx=layer_idx,
            batch_indices=batch_indices,
            new_keys=key_states.transpose(1, 2),
            new_values=value_states.transpose(1, 2),
        )
        # Execution router to route between prefill and decode
        if q_length > 1:
            # # prefill mode, we will use pytorch sdpa
            # num_repeats = self.num_heads // self.num_kv_heads
            # k_contig = key_states.repeat_interleave(num_repeats, dim=2)
            # v_contig = value_states.repeat_interleave(num_repeats, dim=2)

            # SDPA expects shape: [batch_size, num_heads, seq_length, head_dim]
            q_sdpa = query_states.transpose(1, 2)
            k_sdpa = key_states.transpose(1, 2)
            v_sdpa = value_states.transpose(1, 2)

            # Execute using pytorch scaled dot product attention with causal masking
            attn_output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=True, enable_gqa=True
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # decode mode, we will use custom paged attention kernel
            # expects a 3D Query tensor: [batch_size, num_heads, head_dim]
            query_states = query_states.squeeze(1)

            # Slice the tracking tensors to match the query rows
            active_block_tables = block_tables[batch_indices]
            active_context_length = context_length[batch_indices] + q_length

            attn_output = launch_paged_attention(
                query_states,
                kv_cache.key_cache[layer_idx],
                kv_cache.value_cache[layer_idx],
                active_block_tables,
                active_context_length,
            ).unsqueeze(1)
        attn_output = attn_output.view(batch_size, q_length, self.hidden_size)
        return self.o_proj(attn_output)


class Qwen2MLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        # define gating, up and down projection layers for the MLP
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        from tinyinfra.engine.quant.linear import QuantLinear
        if isinstance(self.gate_proj, QuantLinear):
            # Quantized path: each projection dispatches to its Triton dequant GEMM.
            h = x.view(-1, x.shape[-1])
            g = torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h)
            out = self.down_proj(g) + residual.view(-1, residual.shape[-1])
            return out.view(*x.shape[:-1], self.down_proj.out_features)
        # fp16 fused path: gate/up GEMM + SwiGLU + down GEMM + residual
        x_2d = x.view(-1, x.shape[-1])
        residual_2d = residual.view(-1, residual.shape[-1])

        # Single fused pass: gate/up GEMM + SwiGLU + down GEMM + residual
        out_2d = fused_mlp_forward(
            x_2d,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            residual_2d,
        )
        return out_2d.view(*x.shape[:-1], self.down_proj.out_features)


class Qwen2DecodeLayer(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_id: torch.tensor,
        layer_idx: int,
        batch_indices: torch.tensor,
        kv_cache: "KVCacheManager",
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:

        # save a copy of the input hidden states for residual connection
        residual = hidden_states
        # Apply RMS normalization to the input hidden states
        hidden_states = self.input_layernorm(hidden_states)
        # Apply self-attention mechanism to the normalized hidden states
        hidden_states = self.self_attn(
            hidden_states, pos_id, layer_idx, batch_indices, kv_cache, cos_cache, sin_cache
        )

        # Add residual and normalize in one kernel
        hidden_states, residual = fused_add_rmsnorm(
            hidden_states, residual, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        # Apply the MLP to the normalized hidden states (fused down projection + residual)
        hidden_states = self.mlp(hidden_states, residual)
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        # Initialize the embedding layer for input tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Create a list of decoder layers based on the number of layers specified in the configuration
        self.layers = nn.ModuleList([Qwen2DecodeLayer(config) for _ in range(config.num_layers)])
        # Initialize the final RMS normalization layer
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        batch_indices: torch.Tensor,
        kv_cache: "KVCacheManager",
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:

        # extrat the seq_len and batch_size from the input_ids tensor
        seq_len = input_ids.size(1)
        batch_size = input_ids.size(0)

        start_positions = kv_cache.seq_len[batch_indices]

        # Create a 2D pos_id tensor [batch_size, seq_len] offset by the cache length
        base_positions = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        )
        pos_id = base_positions + start_positions.unsqueeze(1)

        hidden_states = self.embed_tokens(input_ids)

        for layer_idx, layers in enumerate(self.layers):
            # Pass the hidden states through each decoder layer, along with position ids and attention mask
            hidden_states = layers(
                hidden_states, pos_id, layer_idx, batch_indices, kv_cache, cos_cache, sin_cache
            )
        # Apply the final RMS normalization to the output hidden states
        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.model = Qwen2Model(config)
        # Initialize the final linear layer for language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        batch_indices: torch.Tensor,
        kv_cache: "KVCacheManager",
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        # Pass the input_ids through the Qwen2 model to obtain hidden states
        hidden_states = self.model(input_ids, batch_indices, kv_cache, cos_cache, sin_cache)
        # Project the hidden states to the vocabulary size using the lm_head
        logits = self.lm_head(hidden_states)
        return logits
