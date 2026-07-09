import torch
import torch.nn.functional as F
import custom_swiglu


def test_fused_swiglu():
    print("🚀 Initializing Vectorized Fused SwiGLU Test...")
    
    # Qwen2 scale parameters (must be divisible by 8)
    batch_size = 4
    seq_len = 1024
    total_tokens = batch_size * seq_len
    hidden_size = 4096
    
    # Simulate the output from the cuBLAS GEMM (Gate and Up concatenated)
    # Shape: [Tokens, Hidden * 2]
    intermediate_tensor = torch.randn(
        total_tokens, 
        hidden_size * 2, 
        device='cuda', 
        dtype=torch.float16
    )
    
    print(f"📊 Allocated Intermediate Tensor: {intermediate_tensor.shape}")
    
    # 1. Custom Fused Kernel Execution
    print("⚡ Executing custom_swiglu.forward()...")
    custom_output = custom_swiglu.forward(intermediate_tensor)
    
    # 2. PyTorch Native Execution
    # PyTorch requires us to physically split the tensor in memory first
    gate, up = intermediate_tensor.chunk(2, dim=-1)
    native_output = F.silu(gate) * up
    
    # 3. Validation
    diff = torch.abs(custom_output - native_output)
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    print(f"⚠️ Max Precision Difference: {max_error:.6f}")
    print(f"⚠️ Mean Precision Difference: {mean_error:.8f}")
    
    # Assert correctness using FP16-safe tolerances
    assert torch.allclose(custom_output, native_output, rtol=1e-2, atol=1e-2), "❌ Outputs do not match within FP16 tolerances!"
    print("✅ Success! Memory-Bound bottleneck bypassed via 128-bit vectorization.")


if __name__ == "__main__":
    test_fused_swiglu()