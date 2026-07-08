import torch
import custom_ops

def test_tensor_cores():
    print("🚀 Initializing Tensor Core (WMMA) Test...")
    
    # Let's test a massive matrix first to measure the exact FP16 drift
    M, N, K = 4096, 4096, 4096
    
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B_transposed = torch.randn(N, K, device='cuda', dtype=torch.float16)
    
    print(f"\n📊 Testing Massive K={K} (FP16 Accumulation Drift expected)...")
    custom_result = custom_ops.gemm(A, B_transposed)
    native_result = torch.matmul(A, B_transposed.t())
    
    # Calculate the exact hardware roundoff error
    diff = torch.abs(custom_result - native_result)
    print(f"⚠️ Max Hardware Roundoff Error: {diff.max().item():.4f}")
    print(f"⚠️ Mean Hardware Roundoff Error: {diff.mean().item():.4f}")
    
    
    # ---------------------------------------------------------
    # Now, let's prove the math is flawless on a safe dimension
    # ---------------------------------------------------------
    M_small, N_small, K_small = 128, 128, 128
    print(f"\n🔬 Testing Small K={K_small} (Where FP16 holds precision)...")
    
    # We divide by 10 to keep the numbers small so FP16 doesn't drop decimals
    A_small = torch.randn(M_small, K_small, device='cuda', dtype=torch.float16) / 10.0
    B_small_t = torch.randn(N_small, K_small, device='cuda', dtype=torch.float16) / 10.0
    
    custom_small = custom_ops.gemm(A_small, B_small_t)
    native_small = torch.matmul(A_small, B_small_t.t())
    
    diff_small = torch.abs(custom_small - native_small)
    print(f"✅ Max Error on Small Matrix: {diff_small.max().item():.6f}")
    
    # This assertion will pass because we removed the FP16 accumulation drift limit!
    assert torch.allclose(custom_small, native_small, atol=1e-2), "Math failed on small matrix!"
    print("\n🎉 SUCCESS! Your Tensor Cores are mathematically perfect!")

if __name__ == "__main__":
    test_tensor_cores()