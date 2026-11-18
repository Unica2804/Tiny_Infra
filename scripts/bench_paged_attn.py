"""Offline tuning bench for the paged-attention kernel (dummy tensors, no checkpoints).

Exercises @triton.autotune with realistic decode shapes: GQA 14 q / 2 kv heads,
head_dim 64, fp16, and reports the config the autotuner selects plus steady-state
microsecond latency per config. Safe to run on 4 GB GPUs (all shapes are small).
"""

import torch
import triton

from tinyinfra.kernels.Paged_Attn import launch_paged_attention, paged_attention_kernel

torch.manual_seed(0)


def make_dummy(batch, ctx, block_size=16, dtype=torch.float16,
               Hq=14, Hkv=2, D=64, q_len=1):
    blocks_per_row = -(-ctx // block_size)
    phys = batch * blocks_per_row
    q = torch.randn(batch, q_len, Hq, D, device="cuda", dtype=dtype)
    k_cache = torch.randn(phys, Hkv, block_size, D, device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.arange(phys, device="cuda", dtype=torch.int32).view(batch, blocks_per_row)
    context_length = torch.full((batch,), ctx + (q_len - 1), device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, block_tables, context_length


def bench_one(name, q, k_cache, v_cache, block_tables, context_length, rep=100):
    fn = lambda: launch_paged_attention(q, k_cache, v_cache, block_tables, context_length)
    ms = triton.testing.do_bench(fn, warmup=25, rep=rep)
    print(f"  {name:34s} {ms * 1e3:8.2f} us")
    return ms


shapes = [
    ("bsz=1  ctx=128 ", (1, 128), 16),
    ("bsz=4  ctx=128 ", (4, 128), 16),
    ("bsz=8  ctx=128 ", (8, 128), 16),
    ("bsz=1  ctx=512 ", (1, 512), 16),
    ("bsz=4  ctx=512 ", (4, 512), 16),
    ("bsz=8  ctx=512 ", (8, 512), 16),
    ("bsz=1  ctx=1024", (1, 1024), 16),
    ("bsz=4  ctx=1024", (4, 1024), 16),
    ("bsz=8  ctx=1024", (8, 1024), 16),
]

print("=== decode shapes (14 q / 2 kv heads, head_dim 64, block 16) ===")
for name, (bsz, ctx), bs in shapes:
    q, kc, vc, bt, cl = make_dummy(bsz, ctx, block_size=bs)
    bench_one(name, q, kc, vc, bt, cl)

print("\n=== block size sweep (bsz=4, ctx=1024) ===")
for bs in (16, 32, 64):
    q, kc, vc, bt, cl = make_dummy(4, 1024, block_size=bs)
    bench_one(f"block_size={bs}", q, kc, vc, bt, cl)

print("\n=== autotuner cache (key -> selected config) ===")
for key, cfg in paged_attention_kernel.cache.items():
    print(f"  key={key}")
    print(f"       -> num_warps={cfg.num_warps}, num_stages={cfg.num_stages}")
