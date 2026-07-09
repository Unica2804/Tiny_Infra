import torch
import torch.nn.functional as F
import custom_swiglu

# Setup problem dimensions (Qwen2 scale)
BATCH_SIZE = 8
SEQ_LEN = 1024
TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
HIDDEN_SIZE = 4096

# Allocate contiguous tensor [Tokens, Hidden * 2]
X = torch.randn(TOTAL_TOKENS, HIDDEN_SIZE * 2, device='cuda', dtype=torch.float16)

# Warm-up iterations to stabilize GPU clocks
for _ in range(20):
    _ = custom_swiglu.forward(X)
    gate, up = X.chunk(2, dim=-1)
    _ = F.silu(gate) * up

# Measure PyTorch Native
start_native = torch.cuda.Event(enable_timing=True)
end_native = torch.cuda.Event(enable_timing=True)

start_native.record()
for _ in range(100):
    gate, up = X.chunk(2, dim=-1)
    native_out = F.silu(gate) * up
end_native.record()
torch.cuda.synchronize()

# Measure Custom Fused Vectorized Kernel
start_custom = torch.cuda.Event(enable_timing=True)
end_custom = torch.cuda.Event(enable_timing=True)

start_custom.record()
for _ in range(100):
    custom_out = custom_swiglu.forward(X)
end_custom.record()
torch.cuda.synchronize()

# Calculate mean execution times
native_time = start_native.elapsed_time(end_native) / 100
custom_time = start_custom.elapsed_time(end_custom) / 100
speedup = native_time / custom_time

print("--- 📊 SwiGLU Performance Results ---")
print(f"Standard PyTorch Time : {native_time * 1000:.2f} µs")
print(f"Custom Fused Kernel   : {custom_time * 1000:.2f} µs")
print(f"Speedup Factor        : {speedup:.2f}x")