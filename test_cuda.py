import torch
import custom_rope

def get_pytorch_rope(q, k, cos, sin, pos_ids):
    """The memory-heavy PyTorch reference implementation."""
    # Fetch the exact cos/sin for the current positions
    cos_sliced = cos[pos_ids].unsqueeze(1) # [Tokens, 1, Half_Dim]
    sin_sliced = sin[pos_ids].unsqueeze(1)
    
    # Split the head dimension
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    
    # Mathematical rotation
    rotated_q = torch.cat((-q2, q1), dim=-1)
    rotated_k = torch.cat((-k2, k1), dim=-1)
    
    # PyTorch allocates massive new tensors right here:
    q_out = (q * cos_sliced.repeat(1, 1, 2)) + (rotated_q * sin_sliced.repeat(1, 1, 2))
    k_out = (k * cos_sliced.repeat(1, 1, 2)) + (rotated_k * sin_sliced.repeat(1, 1, 2))
    
    return q_out, k_out

def test_rope():
    print("🚀 Initializing In-Place RoPE Test...")
    
    # Qwen2 Scale Dimensions
    num_tokens = 1024
    num_heads = 32
    num_kv_heads = 8 # Grouped Query Attention
    head_dim = 128
    half_dim = head_dim // 2
    max_seq_len = 2048
    
    # Initialize inputs
    q = torch.randn(num_tokens, num_heads, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    
    # Cos/Sin tables are usually FP32 for precision
    cos = torch.randn(max_seq_len, half_dim, device='cuda', dtype=torch.float32)
    sin = torch.randn(max_seq_len, half_dim, device='cuda', dtype=torch.float32)
    
    # Fake position IDs (e.g., token 0 to 1023)
    pos_ids = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
    
    # 1. Run PyTorch Reference (Using copies so we don't pollute the original)
    q_ref, k_ref = get_pytorch_rope(q.clone(), k.clone(), cos, sin, pos_ids)
    
    # 2. Run Custom In-Place Kernel
    # Notice we don't assign this to a new variable. The memory is mutated.
    print("⚡ Executing custom_rope.apply_inplace()...")
    custom_rope.apply_inplace(q, k, cos, sin, pos_ids)

    q_ref = q_ref.half()
    k_ref = k_ref.half()
    
    # 3. Validation
    q_diff = torch.abs(q - q_ref).max().item()
    k_diff = torch.abs(k - k_ref).max().item()
    
    print(f"⚠️ Max Q Precision Difference: {q_diff:.6f}")
    print(f"⚠️ Max K Precision Difference: {k_diff:.6f}")
    
    assert torch.allclose(q, q_ref, atol=1e-2, rtol=1e-2), "❌ Q Math mismatch!"
    assert torch.allclose(k, k_ref, atol=1e-2, rtol=1e-2), "❌ K Math mismatch!"
    
    print("✅ Success! Zero-allocation RoPE kernel verified.")

if __name__ == "__main__":
    test_rope()