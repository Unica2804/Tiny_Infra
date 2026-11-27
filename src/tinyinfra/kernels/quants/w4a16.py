import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 2026-11-28 sweep winners for 3B int4 shape
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 64,  'BK': 128, 'GM': 8}, num_stages=2, num_warps=8),
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=2, num_warps=4),
        triton.Config({'BM': 16, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 32, 'BN': 64,  'BK': 64, 'GM': 8}, num_stages=3, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
    ],
    key=['MB', 'N', 'K'],
)

@triton.jit
def w4a16_kernel(
    x_ptrs, qw_ptrs, qz_ptrs, scales_ptrs, y_ptrs,
    M, N, K,
    MB: tl.constexpr,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzk, stride_qzn,
    stride_sk, stride_sn,
    stride_ym, stride_yn,
    group_sz: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_programs_m = tl.cdiv(M, BM)
    num_programs_n = tl.cdiv(N, BN)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_programs_m, num_programs_n, GM)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_col = offs_n < N

    x_ptrs = x_ptrs + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_offset = k * BK
        k_idx = k_offset + offs_k
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)

        # Simpler correct unpack: gather per-K word and shift (1 load per K row).
        w_each = tl.load(qw_ptrs + (k_idx // 8)[:, None] * stride_qwk
                         + offs_n[None, :] * stride_qwn,
                         mask=(k_idx[:, None] < K) & mask_col[None, :], other=0)
        w_i4 = ((w_each >> ((k_idx % 8)[:, None] * 4)) & 0x0F).to(tl.float16)

        # qzeros [K/g, N/8]: group-major, 8 nibbles packed along N.
        group_idx = k_offset // group_sz
        z_each = tl.load(qz_ptrs + group_idx * stride_qzk
                         + (offs_n // 8)[None, :] * stride_qzn,
                         mask=mask_col[None, :], other=0)
        z_i4 = ((z_each >> ((offs_n % 8)[None, :] * 4)) & 0x0F).to(tl.float16)

        s = tl.load(scales_ptrs + group_idx * stride_sk + offs_n[None, :] * stride_sn,
                    mask=mask_col[None, :], other=0.0)          # fp16 [1,BN]
        w_fp16 = (w_i4 - z_i4) * s
        acc = tl.dot(x, w_fp16, acc)

        x_ptrs += BK * stride_xk

    y_ptrs = y_ptrs + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & mask_col[None, :])


def w4a16_matmul(X: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
                 scales: torch.Tensor, group_sz: int = 128) -> torch.Tensor:
    M, K = X.shape
    K_packed, N = qweight.shape
    assert K == K_packed * 8, f"K ({K}) must equal qweight.shape[0]*8 ({K_packed*8})"
    assert K % group_sz == 0, "K must be divisible by group_sz"
    y = torch.empty((M, N), device=X.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']))
    w4a16_kernel[grid](
        X, qweight, qzeros, scales, y,
        M, N, K,
        (0 if M <= 16 else 1),
        X.stride(0), X.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        group_sz=group_sz,
    )
    return y
