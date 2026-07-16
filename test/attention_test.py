import torch
import torch.nn.functional as F
import custom_paged_attn
from engine.cache_manager import KVCacheManager

def gather_contiguous_kv(k_cache, v_cache, block_table, context_len, block_size, num_kv_heads, head_dim):
    """Reconstructs the fragmented VRAM pool into a contiguous tensor for PyTorch validation."""
    k_contiguous = torch.zeros((num_kv_heads, context_len, head_dim), dtype=torch.float16, device='cuda')
    v_contiguous = torch.zeros((num_kv_heads, context_len, head_dim), dtype=torch.float16, device='cuda')
    
    for i in range(context_len):
        logical_block = i // block_size
        physical_block = block_table[logical_block].item()
        block_offset = i % block_size
        
        # k_cache shape is [physical_blocks, num_kv_heads, block_size, head_dim]
        k_contiguous[:, i, :] = k_cache[physical_block, :, block_offset, :]
        v_contiguous[:, i, :] = v_cache[physical_block, :, block_offset, :]
        
    return k_contiguous, v_contiguous

def test_full_engine_integration():
    print("🚀 Initializing Full Engine Integration Test...")

    # Architecture
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2  # Grouped Query Attention (4 Q heads share 1 KV head)
    head_dim = 128
    block_size = 16
    max_seq_len = 128
    layer_idx = 0

    # Initialize your custom memory manager
    manager = KVCacheManager(
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=1, # Testing 1 layer
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        device=torch.device('cuda'),
        dtype=torch.float16
    )

    batch_indices = torch.tensor([0, 1], dtype=torch.int32, device='cuda')

    # =========================================================================
    # PHASE 1: THE PREFILL (Initial Prompt)
    # User 0 sends a 25-token prompt. User 1 sends a 12-token prompt.
    # =========================================================================
    print("📥 Simulating Prefill Phase...")
    prefill_len = 25 
    
    # We pad User 1's prompt to match the tensor shape, but the sequence length tracker will handle the reality
    k_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim, dtype=torch.float16, device='cuda') * 0.1
    v_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim, dtype=torch.float16, device='cuda') * 0.1
    
    manager.allocate_and_insert(layer_idx, batch_indices, k_prefill, v_prefill)
    # Manually correct User 1's seq_len since we simulated a padded prefill
    manager.seq_len[1] = 12 

    # =========================================================================
    # PHASE 2: THE DECODE (Generating 1 Token)
    # Both users generate exactly 1 new token
    # =========================================================================
    print("🔄 Simulating Decode Phase (+1 Token)...")
    k_decode = torch.randn(batch_size, num_kv_heads, 1, head_dim, dtype=torch.float16, device='cuda') * 0.1
    v_decode = torch.randn(batch_size, num_kv_heads, 1, head_dim, dtype=torch.float16, device='cuda') * 0.1

    # This should dynamically cross the block boundary for User 0 (from token 25 to 26)
    block_tables, seq_lens = manager.allocate_and_insert(layer_idx, batch_indices, k_decode, v_decode)

    # =========================================================================
    # PHASE 3: THE ATTENTION KERNEL VALIDATION
    # =========================================================================
    print("⚔️ Executing PagedAttention vs PyTorch Native...")
    
    # Generate the incoming Query vector for the current single decode step
    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device='cuda') * 0.1
    out_custom = torch.zeros_like(q)

    # --- 1. Run Custom C++ PagedAttention ---
    custom_paged_attn.launch_paged_attention(
        out_custom, 
        q, 
        manager.key_cache[layer_idx],   # Pass only the 4D slice for this specific layer
        manager.value_cache[layer_idx], 
        block_tables, 
        seq_lens, 
        manager.block_size
    )

    # --- 2. Run PyTorch Contiguous Reference ---
    out_ref = torch.zeros_like(q)
    for i in range(batch_size):
        c_len = seq_lens[i].item()
        
        k_contig, v_contig = gather_contiguous_kv(
            manager.key_cache[layer_idx], manager.value_cache[layer_idx], 
            block_tables[i], c_len, block_size, num_kv_heads, head_dim
        )

        num_repeats = num_heads // num_kv_heads
        k_contig = k_contig.repeat_interleave(num_repeats, dim=0)
        v_contig = v_contig.repeat_interleave(num_repeats, dim=0)
        
        # Reshape for SDPA
        q_i = q[i].unsqueeze(1) 
        attn_out = F.scaled_dot_product_attention(q_i, k_contig, v_contig)
        out_ref[i] = attn_out.squeeze(1)

    # --- 3. Compare ---
    diff = torch.abs(out_custom - out_ref).max().item()
    print(f"⚠️ Max Precision Difference: {diff:.6f}")

    assert torch.allclose(out_custom, out_ref, atol=1e-2, rtol=1e-2), "❌ Math mismatch in Integration!"
    print("✅ Success! Manager successfully routed the memory to the custom kernel.")

if __name__ == "__main__":
    test_full_engine_integration()