import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Sweep winners 2026-11-27 (BM16/BN128/w4/s2 decode, BM64/BN128/w4/s3 prefill)
        triton.Config({'BM': 16, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=4),
        # Seeded from the w8a16 sweep; W8A8 uses int8 tensor-core dot (int32 accumulate).
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        triton.Config({'BM': 32, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=3, num_warps=4),
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 16, 'BN': 128, 'BK': 128, 'GM': 8}, num_stages=3, num_warps=4),
    ],
    key=['MB', 'N', 'K'],   # MB = M bucket (decode/prefill), not exact length
)

@triton.jit
def w8a8_kernel(
    x_ptrs, w_ptrs, xscales_ptrs, wscales_ptrs, y_ptrs,
    M, N, K,
    MB: tl.constexpr,          # 0 = small M (decode), 1 = large M (prefill)
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_xs,
    stride_ws_group, stride_ws_n,
    stride_ym, stride_yn,
    group_sz : tl.constexpr,
    BM : tl.constexpr,
    BN : tl.constexpr,
    BK : tl.constexpr,
    GM : tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_programs_m = tl.cdiv(M, BM)
    num_programs_n = tl.cdiv(N, BN)

    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_programs_m, num_programs_n, GM)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    x_ptrs = x_ptrs + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptrs + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_offset = k * BK
        k_idx = k_offset + offs_k
        mask_x = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        mask_w = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        x_i8 = tl.load(x_ptrs, mask=mask_x, other=0)         # int8 activations
        w_i8 = tl.load(w_ptrs, mask=mask_w, other=0)         # int8 weights

        # One quant group per K tile (group_sz % BK == 0 enforced by wrapper).
        group_idx = k_offset // group_sz
        ws_ptrs = wscales_ptrs + (group_idx * stride_ws_group + offs_n[None, :] * stride_ws_n)
        w_scale = tl.load(ws_ptrs, mask=offs_n[None, :] < N, other=0.0).to(tl.float32)

        # int8 x int8 -> int32 tensor-core dot; dequantize the partial with the weight scale.
        partial = tl.dot(x_i8, w_i8, out_dtype=tl.int32)      # [BM, BN] int32
        acc += partial.to(tl.float32) * w_scale

        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk

    # Activation scale is per-token (row); weight scale already applied per group.
    x_scale = tl.load(xscales_ptrs + offs_m * stride_xs, mask=offs_m < M, other=0.0).to(tl.float32)
    out = acc * x_scale[:, None]

    y_ptrs = y_ptrs + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, out.to(tl.float16), mask=mask_y)


def quantize_int8(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token int8 quantization (symmetric, no zero-point)."""
    scale = X.float().abs().amax(dim=1) / 127.0               # [M]
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_q = torch.round(X.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return x_q, scale.to(torch.float16)


def w8a8_matmul(
    X_q: torch.Tensor,
    W_q: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    group_sz: int = 128,
) -> torch.Tensor:
    """X_q: int8 [M,K]; W_q: int8 [K,N]; x_scales: fp16 [M]; w_scales: fp16 [K/g,N]."""
    M, K = X_q.shape
    K_w, N = W_q.shape
    assert K == K_w, f"X_q.shape[1] ({K}) must match W_q.shape[0] ({K_w})"
    assert X_q.dtype == torch.int8 and W_q.dtype == torch.int8, "W8A8 requires int8 inputs"
    assert (group_sz % 64 == 0 or 64 % group_sz == 0), "group_sz must align with 64"
    assert K % group_sz == 0, "K must be divisible by group_sz"
    y = torch.empty((M, N), device=X_q.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']))

    w8a8_kernel[grid](
        X_q, W_q, x_scales, w_scales, y,
        M, N, K,
        (0 if M <= 16 else 1),
        X_q.stride(0), X_q.stride(1),
        W_q.stride(0), W_q.stride(1),
        x_scales.stride(0),
        w_scales.stride(0), w_scales.stride(1),
        y.stride(0), y.stride(1),
        group_sz=group_sz,
    )
    return y
