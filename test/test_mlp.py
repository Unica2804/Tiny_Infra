import torch
import torch.nn.functional as F
import time
from models.mlp import Qwen2MLP
from engine.config import QwenConfig


def run_reference_mlp(x, fused_weight, down_weight):
    """Reference unfused PyTorch MLP execution path for verification."""
    fused_out = F.linear(x, fused_weight)
    gate, up = fused_out.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    return F.linear(activated, down_weight)


def test_mlp_integration():
    print("🚀 Initializing End-to-End MLP Block Test...")
    config = QwenConfig()
    # Qwen2-7B Hyperparameters
    batch_size = 4
    seq_len = 1024
    tokens = batch_size * seq_len
    
    # Initialize inputs and weights
    x = torch.randn(tokens, config.hidden_size, device='cuda', dtype=torch.float16)
    
    mlp = Qwen2MLP(config).cuda().eval()
    
    # Extract weights to ensure identical baselines for mathematical comparison
    fused_w = mlp.fused_gate_up_proj.weight
    down_w = mlp.down_proj.weight

    # Warm-up cycles to stabilize GPU clocks
    for _ in range(10):
        _ = mlp(x)
        _ = run_reference_mlp(x, fused_w, down_w)
        
    # 1. Correctness Validation
    custom_mlp_out = mlp(x)
    reference_mlp_out = run_reference_mlp(x, fused_w, down_w)
    
    # Standard FP16 validation setting using relative and absolute tolerances
    assert torch.allclose(custom_mlp_out, reference_mlp_out, rtol=1e-2, atol=1e-2), "❌ Math mismatch in block integration!"
    print("✅ Math Verification: Passed! Integrated custom kernel matches PyTorch perfectly.")

    # 2. Performance Profiling
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(100):
        _ = run_reference_mlp(x, fused_w, down_w)
    torch.cuda.synchronize()
    reference_duration = (time.perf_counter() - start_time) / 100

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(100):
        _ = mlp(x)
    torch.cuda.synchronize()
    custom_duration = (time.perf_counter() - start_time) / 100

    print("\n--- 📊 End-to-End MLP Layer Performance ---")
    print(f"Reference PyTorch MLP Block: {reference_duration * 1000:.3f} ms")
    print(f"Custom Fused MLP Block     : {custom_duration * 1000:.3f} ms")
    print(f"Overall Block Speedup      : {reference_duration / custom_duration:.2f}x")


if __name__ == "__main__":
    test_mlp_integration()