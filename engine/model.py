# Architecture for Qwen-2.5-0.5B. It uses the Qwen 2 Architecture
# No use of fused operations, as it will be optimized using custom kernels for rtx2050 mobile gpu.
# All operations used are primitive pytorch ops or linear layers, which can be fused later on.

import torch
import torch.nn as nn
from config import QwenConfig
import math
from engine.cache_manager import KVCacheManager 



class RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> torch.Tensor:
        
        super().__init__()
        # Initialize the weight parameter for the RMS normalization layer
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Store the epsilon value for numerical stability
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor, but got {type(hidden_states)}")
        # calculation must happen in float32 for numerical stability hence store oriiginal
        input_dtype = hidden_states.dtype
        # Cast to float32 for numerical stability during RMS normalization
        hidden_states = hidden_states.to(torch.float32)
        # Compute the variance of the hidden states along the last dimension
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        # Normalize the hidden states using RMS normalization formula
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # Scale the normalized hidden states by the weight parameter
        return self.weight * hidden_states.to(input_dtype)  # Cast back to original dtype

# helper for rotary positional embedding
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError(f"Expected q and k to be torch.Tensors, but got {type(q)} and {type(k)}")
    if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        raise TypeError(f"Expected cos and sin to be torch.Tensors, but got {type(cos)} and {type(sin)}")
    # split the last dimension of q and k into two halves
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    # Rotate the second half negatively for q and k
    q_rotated = torch.cat([-q2, q1], dim=-1)
    k_rotated = torch.cat([-k2, k1], dim=-1)
    # Apply the rotary positional embeddings using cosine and sine components
    q_embed = (q * cos) + (q_rotated * sin)
    k_embed = (k * cos) + (k_rotated * sin)
    return q_embed, k_embed

class Qwen2Attention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor, pos_id: torch.tensor, attention_mask: torch.Tensor, layer_idx: int,
        batch_indices: torch.Tensor, kv_cache: "KVCacheManager") -> torch.Tensor:
        # Extract the batch size, sequence length from the input hidden states
        batch_size, q_length, _ = hidden_states.size()

        # Project the hidden states to query, key and value matrices.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Transform the query, key, and value matrices to the shape (batch_size, num_heads, seq_length, head_dim)
        query_states = query_states.view(batch_size, q_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Rope and kernel goes here it will be optimized using custom kernels for rtx2050 mobile gpu.
        #
        #
        #
        
        # Handover the cached key and value states to the KVCacheManager for updating and fetching
        key_states, value_states = kv_cache.update_and_fetch(
            layer_idx = layer_idx,
            batch_indices = batch_indices,
            new_keys = key_states,
            new_values = value_states
        )
        
        # match number of heads for query and key/value by repeating key and value states for gqa
        key_states = torch.repeat_interleave(key_states, repeats=self.num_heads // self.num_kv_heads, dim=1)
        value_states = torch.repeat_interleave(value_states, repeats=self.num_heads // self.num_kv_heads, dim=1)

        # Compute the attention scores using scaled dot-product attention
        attention_scores = torch.matmul(query_states, key_states.transpose(2,3))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        #Ensure the attenstion mask matches the shape of the previous attention scores
        # slice the attention mask dynamically to match the current sequence length
        seq_len = key_states.size(-2)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask[:,:,:q_length,:seq_len]

        # Apply softmax to obtain attention probabilities
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1,dtype=torch.float32).to(query_states.dtype)

        # Compute the context layer by multiplying attention probabilities with value states
        context_layer = torch.matmul(attention_scores, value_states)

        # Reshape the context layer to the original hidden size
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, q_length, self.hidden_size)

        # Project the context layer back to the hidden size
        return self.o_proj(context_layer)

class Qwen2MLP(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        # define gating, up and down projection layers for the MLP
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Define Swilu activation function for the MLP
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the gated activation using the gate projection and Swilu activation function
        gated_states = self.act_fn(self.gate_proj(x))
        # Compute the up projection of the input
        up_states = self.up_proj(x)
        # Element-wise multiplication of gated activation and up projection
        hidden_states = gated_states * up_states
        # Project the hidden states back to the original hidden size
        return self.down_proj(hidden_states)

class Qwen2DecodeLayer(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor, pos_id: torch.tensor,
        attention_mask: torch.Tensor, layer_idx: int, batch_indices: torch.tensor,kv_cache: "KVCacheManager") -> torch.Tensor:
        # save a copy of the input hidden states for residual connection
        residual = hidden_states
        # Apply RMS normalization to the input hidden states
        hidden_states = self.input_layernorm(hidden_states)
        # Apply self-attention mechanism to the normalized hidden states
        hidden_states = self.self_attn(hidden_states, pos_id, attention_mask,layer_idx, batch_indices, kv_cache)
        # Add the residual connection to the output of self-attention
        hidden_states = hidden_states + residual

        # save a new copy of the hidden states for the next residual connection
        residual = hidden_states
        # Apply RMS normalization to the hidden states after self-attention
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Apply the MLP to the normalized hidden states
        hidden_states = self.mlp(hidden_states)
        # Add the residual connection to the output of the MLP
        hidden_states = hidden_states + residual
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

    def _create_attention_mask(self, seq_len: int, device: torch.device)-> torch.Tensor:
        # Create a square attention mask with shape (seq_len, seq_len) filled with -inf
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        # Set the lower triangular part of the mask to 0, allowing attention to previous tokens
        mask = torch.triu(mask, diagonal=1)
        # Add batch and head dimensions
        return mask.unsqueeze(0).unsqueeze(0)  
    
    def forward(self, input_ids: torch.Tensor,batch_indices: torch.Tensor,kv_cache: "KVCacheManager") -> torch.Tensor:
        # if input_ids not in (torch.int64, torch.int32):
        #     raise TypeError(f"Expected input_ids to be of type torch.int64 or torch.int32, but got {input_ids.dtype}")
        # extrat the seq_len from the input_ids tensor
        seq_len = input_ids.size(1)
        # generate a sequence of position ids for the input tokens
        pos_id = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # Look up the embeddings for the input tokens using the embedding layer
        hidden_states = self.embed_tokens(input_ids)
        # Create the attention mask for the input sequence
        attention_mask = self._create_attention_mask(seq_len, hidden_states.device)

        for layer_idx, layers in enumerate(self.layers):
            # Pass the hidden states through each decoder layer, along with position ids and attention mask
            hidden_states = layers(hidden_states, pos_id, attention_mask, layer_idx, batch_indices, kv_cache)
        # Apply the final RMS normalization to the output hidden states
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.model = Qwen2Model(config)
        # Initialize the final linear layer for language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, batch_indices: torch.Tensor, kv_cache: "KVCacheManager") -> torch.Tensor:
        # Pass the input_ids through the Qwen2 model to obtain hidden states
        hidden_states = self.model(input_ids, batch_indices, kv_cache)
        # Project the hidden states to the vocabulary size using the lm_head
        logits = self.lm_head(hidden_states)
        return logits