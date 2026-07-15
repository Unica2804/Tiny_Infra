import torch
import torch.nn.functional as F
import custom_paged_attn
import math

def gather_contiguous_kv(k_cache, v_cache, block_table, context_len, block_size, num_kv_heads, head_dim):
    """
    Simulates the standard PyTorch requirement: 
    Reconstructs a fragmented paged cache back into a single contiguous tensor.
    """
    k_contiguous = torch.zeros((num_kv_heads, context_len, head_dim), dtype=torch.float16, device='cuda')
    v_contiguous = torch.zeros((num_kv_heads, context_len, head_dim), dtype=torch.float16, device='cuda')
    
    for i in range(context_len):
        logical_block = i // block_size
        physical_block = block_table[logical_block].item()
        block_offset = i % block_size
        
        k_contiguous[:, i, :] = k_cache[physical_block, :, block_offset, :]
        v_contiguous[:, i, :] = v_cache[physical_block, :, block_offset, :]
        
    return k_contiguous, v_contiguous

def test_micro_paged_attention():
    print("🚀 Initializing Micro-PagedAttention Validation...")

    # Architecture Dimensions
    batch_size = 2
    num_heads = 4
    num_kv_heads = 4 # Standard MHA for this test (GQA is supported in C++)
    head_dim = 128
    block_size = 16
    
    # Memory Pool Dimensions
    total_physical_blocks = 20
    max_blocks_per_seq = 4

    # 1. Simulate the Pre-Allocated VRAM Pool (Fragmented Memory)
    k_cache = torch.randn(total_physical_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(total_physical_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device='cuda')

    # 2. Simulate User Requests
    # Sequence 0 has 45 tokens. Sequence 1 has 23 tokens.
    context_lens = torch.tensor([45, 23], dtype=torch.int32, device='cuda')
    
    # 3. Simulate the Block Table (Random physical pages assigned to logical blocks)
    block_tables = torch.tensor([
        [14, 2, 8, 19],  # Seq 0 uses physical pages 14, 2, 8, and 19
        [5, 11, -1, -1]  # Seq 1 uses physical pages 5 and 11
    ], dtype=torch.int32, device='cuda')

    # 4. The current generation step (Query for the exact current token)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device='cuda')
    out_custom = torch.zeros_like(q)

    # --- EXECUTE PYTORCH REFERENCE ---
    print("📊 Computing PyTorch Contiguous Baseline...")
    out_ref = torch.zeros_like(q)
    for i in range(batch_size):
        c_len = context_lens[i].item()
        if c_len == 0: continue
        
        # Reconstruct memory
        k_contig, v_contig = gather_contiguous_kv(
            k_cache, v_cache, block_tables[i], c_len, block_size, num_kv_heads, head_dim
        )
        
        # Reshape for SDPA: [Heads, 1, Head_Dim] and [Heads, Seq, Head_Dim]
        q_i = q[i].unsqueeze(1) 
        k_i = k_contig
        v_i = v_contig
        
        # Math calculation
        attn_out = F.scaled_dot_product_attention(q_i, k_i, v_i)
        out_ref[i] = attn_out.squeeze(1)

    # --- EXECUTE CUSTOM KERNEL ---
    print("⚡ Executing custom_paged_attn.launch_paged_attention()...")
    custom_paged_attn.launch_paged_attention(
        out_custom, q, k_cache, v_cache, block_tables, context_lens, block_size
    )

    # --- VALIDATION ---
    diff = torch.abs(out_custom - out_ref).max().item()
    print(f"⚠️ Max Precision Difference: {diff:.6f}")

    # Standard FP16 Tiling tolerances
    assert torch.allclose(out_custom, out_ref, atol=1e-2, rtol=1e-2), "❌ Math mismatch in PagedAttention!"
    print("✅ Success! Custom PagedAttention kernel matches PyTorch exactly.")

if __name__ == "__main__":
    test_micro_paged_attention()