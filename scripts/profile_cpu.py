"""CPU-side profile: prefill (TTFT) vs decode (throughput), separately.

Usage:
    uv run python scripts/profile_cpu.py --steps 20
    uv run python -m cProfile -s cumtime scripts/profile_cpu.py --steps 5
"""

import argparse
import cProfile
import io
import pstats
import time

import torch

from tinyinfra.engine.cache_manager import KVCacheManager
from tinyinfra.engine.loader import load_model


def build():
    loaded = load_model("weights/qwen", device="cuda", dtype=torch.float16)
    model, cfg = loaded.model, loaded.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    fr = torch.outer(torch.arange(2048, device="cuda", dtype=torch.float32), inv)
    cos, sin = fr.cos().contiguous(), fr.sin().contiguous()
    kv = KVCacheManager(1, 2048, cfg.num_layers, cfg.num_key_value_heads, hd,
                        16, torch.device("cuda"), torch.float16)
    return model, cfg, kv, cos, sin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    model, cfg, kv, cos, sin = build()
    bi = torch.tensor([0], device="cuda", dtype=torch.int32)
    prompt = torch.arange(1, 33, device="cuda", dtype=torch.int32).view(1, -1)
    token = torch.tensor([[42]], device="cuda", dtype=torch.int32)

    # warm kernels so CPU time isn't JIT compilation
    with torch.no_grad():
        model(prompt, bi, kv, cos, sin)
    kv.seq_len[bi] += 32
    kv.free_batch_slot(0)
    with torch.no_grad():
        model(token, bi, kv, cos, sin)
    kv.free_batch_slot(0)
    torch.cuda.synchronize()

    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()

    # --- prefill timing (TTFT proxy) ---
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            model(prompt, bi, kv, cos, sin)
        kv.free_batch_slot(0)
    torch.cuda.synchronize()
    print(f"prefill: {(time.perf_counter()-t0)/10*1000:.2f} ms/call  (TTFT proxy)")

    # --- decode timing ---
    t0 = time.perf_counter()
    for _ in range(args.steps):
        with torch.no_grad():
            model(token, bi, kv, cos, sin)
        kv.seq_len[bi] += 1
    torch.cuda.synchronize()
    print(f"decode:  {(time.perf_counter()-t0)/args.steps*1000:.2f} ms/token")

    if prof:
        prof.disable()
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(40)
        print(s.getvalue())


if __name__ == "__main__":
    main()
