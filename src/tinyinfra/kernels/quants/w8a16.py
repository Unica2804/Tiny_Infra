import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Best read from the 2026-11-24 sweep: decode (M<=4) + prefill (M>=128)
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        # Alternates that won at intermediate M
        triton.Config({'BM': 32, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=3, num_warps=4),
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 16, 'BN': 128, 'BK': 128, 'GM': 8}, num_stages=3, num_warps=4),
    ],
    key=['MB', 'N', 'K'],   # MB = M bucket (decode/prefill), not exact length
)

@triton.jit
def w8a16_kernel(
    x_ptrs, w_ptrs, scales_ptrs, zeros_ptrs, y_ptrs,
    M, N, K,
    MB: tl.constexpr,          # 0 = small M (decode), 1 = large M (prefill)
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_zk, stride_zn,
    stride_ym, stride_yn,
    group_sz : tl.constexpr,
    BM : tl.constexpr, 
    BN : tl.constexpr,
    BK : tl.constexpr,
    GM: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    num_programs_m = tl.cdiv(M, BM)
    num_programs_n = tl.cdiv(N, BN)

    pid_m , pid_n = tl.swizzle2d(pid_m, pid_n, num_programs_m, num_programs_n, GM)

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

        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_u8 = tl.load(w_ptrs, mask=mask_w, other=0)

        group_idx = k_idx // group_sz
        mask_sz = (group_idx[:, None] < tl.cdiv(K, group_sz)) & (offs_n[None, :] < N)

        s_ptrs = scales_ptrs + (group_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn)
        z_ptrs = zeros_ptrs + (group_idx[:, None] * stride_zk + offs_n[None, :] * stride_zn)

        scales = tl.load(s_ptrs, mask=mask_sz, other=0.0)
        zeros = tl.load(z_ptrs, mask=mask_sz, other=0)

        w_fp16 = (w_u8.to(tl.float16) - zeros.to(tl.float16)) * scales

        acc = tl.dot(x, w_fp16, acc)

        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk
    y_ptrs = y_ptrs + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask_y = (offs_m[:, None]<M) & (offs_n[None, :]<N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=mask_y)

def w8a16_matmul(
    X:torch.Tensor,
    W:torch.Tensor,
    scales:torch.Tensor,
    zeros:torch.Tensor,
    group_sz:int=128,
) -> torch.Tensor:
    M, K = X.shape
    K_w, N = W.shape
    assert K == K_w, f"X.shape[1] ({K}) must match W.shape[0] ({K_w})"
    assert (group_sz % 64 == 0 or 64 % group_sz == 0), "group_sz must align with 64"
    y = torch.empty((M, N), device=X.device, dtype=torch.float16)

    
    grid = lambda META: (triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']))

    w8a16_kernel[grid](
        X, W, scales, zeros, y,
        M, N, K,
        (0 if M <= 16 else 1),   # MB bucket, avoids per-prompt-length retuning
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        y.stride(0), y.stride(1),
        group_sz=group_sz,
    )
    return y