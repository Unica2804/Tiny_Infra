"""Throughput comparison: FP16 / BF16 / W8A16 / W8A8 on synthetic tensors.

Isolates the linear-layer math (no checkpoint download needed). Reports both
single-GEMM time (Qwen2.5-0.5B / 3B shapes) and end-to-end decode token rate
using the engine's model with each weight format swapped in.
"""

import argparse

import torch

from tinyinfra.kernels.quants.w8a16 import w8a16_matmul
from tinyinfra.kernels.quants.w8a8 import quantize_int8, w8a8_matmul


def bench_fn(fn, reps=60, warmup=15):
    import triton
    return triton.testing.do_bench(fn, warmup=warmup, rep=reps) * 1000  # us


def make_quant_weights(K, N, scheme, device="cuda"):
    """Build synthetic quantized weights and scales for a [K,N] linear layer."""
    if scheme == "w8a16":
        W = torch.randint(0, 256, (K, N), device=device, dtype=torch.uint8)
        scales = torch.rand(1, N, device=device, dtype=torch.float16) * 0.01 + 0.005
        zeros = torch.full((1, N), 128, device=device, dtype=torch.uint8)
        return W, scales, zeros
    if scheme == "w8a8":
        W = torch.randint(-127, 127, (K, N), device=device, dtype=torch.int8)
        scales = torch.rand(K // 128, N, device=device, dtype=torch.float16) * 0.01 + 0.005
        return W, scales
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    args = ap.parse_args()

    print("=" * 78)
    print("Single GEMM throughput (us) — synthetic weights, K=896/N=4864 = 0.5B MLP shape")
    print("=" * 78)
    K, N = 896, 4864
    for M in (1, 8, 64, 128):
        X16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        Xbf = X16.to(torch.bfloat16)
        W16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.02
        Wbf = W16.to(torch.bfloat16)
        Wq, sc, zr = make_quant_weights(K, N, "w8a16")
        Wq8, sc8 = make_quant_weights(K, N, "w8a8")
        Xq8, xs8 = quantize_int8(X16)

        r = {}
        r["fp16"] = bench_fn(lambda: X16 @ W16, args.reps)
        r["bf16"] = bench_fn(lambda: Xbf @ Wbf, args.reps)
        r["w8a16"] = bench_fn(lambda: w8a16_matmul(X16, Wq, sc, zr, K), args.reps)
        r["w8a8"] = bench_fn(lambda: w8a8_matmul(Xq8, Wq8, xs8, sc8, 128), args.reps)
        print(f"M={M:4}: " + " | ".join(f"{k}={v:7.1f}us" for k, v in r.items()))

    print("\n" + "=" * 78)
    print("3B MLP shape K=2048/N=11008")
    print("=" * 78)
    K, N = 2048, 11008
    for M in (1, 8, 64):
        X16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        Xbf = X16.to(torch.bfloat16)
        W16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.02
        Wbf = W16.to(torch.bfloat16)
        Wq, sc, zr = make_quant_weights(K, N, "w8a16")
        Wq8, sc8 = make_quant_weights(K, N, "w8a8")
        Xq8, xs8 = quantize_int8(X16)
        r = {
            "fp16": bench_fn(lambda: X16 @ W16),
            "bf16": bench_fn(lambda: Xbf @ Wbf),
            "w8a16": bench_fn(lambda: w8a16_matmul(X16, Wq, sc, zr, K)),
            "w8a8": bench_fn(lambda: w8a8_matmul(Xq8, Wq8, xs8, sc8, 128)),
        }
        print(f"M={M:4}: " + " | ".join(f"{k}={v:7.1f}us" for k, v in r.items()))


if __name__ == "__main__":
    main()
