"""Isolated 1-token decode workload for nsys / ncu profiling.

Usage:
    uv run python scripts/profile_decode.py           # warmup only, measures baseline
    nsys profile -o /tmp/opencode/nsys_decode --force-overwrite true \
        uv run python scripts/profile_decode.py --warmup 2 --steps 40
    ncu --set full --kernel-name regex:... -o /tmp/opencode/ncu_decode \
        uv run python scripts/profile_decode.py --warmup 1 --steps 1

The script warms up all kernels (avoiding JIT/autotune in the measured region),
then runs N decode steps with a barrier between each so ncu attributes each
kernel launch to a single step.
"""

import argparse
import time

import torch

from tinyinfra.engine.cache_manager import KVCacheManager
from tinyinfra.engine.loader import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    loaded = load_model(args.model, device="cuda", dtype=torch.float16)
    model, cfg = loaded.model, loaded.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)
    )
    freqs = torch.outer(torch.arange(2048, device="cuda", dtype=torch.float32), inv_freq)
    cos_cache, sin_cache = freqs.cos().contiguous(), freqs.sin().contiguous()

    kv = KVCacheManager(
        max_batch_size=1,
        max_seq_len=2048,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        block_size=16,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    batch_indices = torch.tensor([0], device="cuda", dtype=torch.int32)

    # prefill a 32-token prompt so the KV cache has real context
    prompt = torch.arange(1, 33, device="cuda", dtype=torch.int32).view(1, -1)
    with torch.no_grad():
        _ = model(prompt, batch_indices, kv, cos_cache, sin_cache)
    kv.seq_len[batch_indices] += prompt.shape[1]

    token = torch.tensor([[42]], device="cuda", dtype=torch.int32)

    # warm up all Triton JIT kernels + autotune before measurement
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model(token, batch_indices, kv, cos_cache, sin_cache)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        with torch.no_grad():
            out = model(token, batch_indices, kv, cos_cache, sin_cache)
        kv.seq_len[batch_indices] += 1
        torch.cuda.synchronize()  # per-step barrier -> clean ncu/nsys attribution
    dt = (time.perf_counter() - start) / args.steps
    print(f"decode: {dt * 1000:.2f} ms/token -> {1 / dt:.1f} tok/s (incl. sync barrier)")


if __name__ == "__main__":
    main()
