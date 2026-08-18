import triton
import triton.language as tl
import torch


@triton.jit
def fused_gateup_swiglu_kernel(
    X, Wg, Wu, Y,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + offs_m[:, None] * K + offs_k[None, :]
    wg_ptrs = Wg + offs_n[None, :] * K + offs_k[:, None]
    wu_ptrs = Wu + offs_n[None, :] * K + offs_k[:, None]

    x_mask = offs_m[:, None] < M

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        km = offs_k[None, :] < (K - k * BLOCK_K)
        wkm = offs_k[:, None] < (K - k * BLOCK_K)
        xk = tl.load(x_ptrs, mask=x_mask & km, other=0.0)
        wg = tl.load(wg_ptrs, mask=wkm, other=0.0)
        wu = tl.load(wu_ptrs, mask=wkm, other=0.0)
        acc_g = tl.dot(xk, wg, acc_g)
        acc_u = tl.dot(xk, wu, acc_u)
        x_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K
        wu_ptrs += BLOCK_K

    gate = acc_g.to(tl.float32)
    up = acc_u.to(tl.float32)
    res = (gate * tl.sigmoid(gate)) * up

    y_ptrs = Y + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, res.to(X.dtype.element_ty), mask=x_mask & (offs_n[None, :] < N))


@triton.jit
def fused_down_residual_kernel(
    S, Wd, R, Y,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    s_ptrs = S + offs_m[:, None] * K + offs_k[None, :]
    wd_ptrs = Wd + offs_n[None, :] * K + offs_k[:, None]

    x_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        km = offs_k[None, :] < (K - k * BLOCK_K)
        wkm = offs_k[:, None] < (K - k * BLOCK_K)
        sk = tl.load(s_ptrs, mask=x_mask & km, other=0.0)
        wd = tl.load(wd_ptrs, mask=wkm, other=0.0)
        acc = tl.dot(sk, wd, acc)
        s_ptrs += BLOCK_K
        wd_ptrs += BLOCK_K

    r_ptrs = R + offs_m[:, None] * N + offs_n[None, :]
    resid = tl.load(r_ptrs, mask=x_mask & (offs_n[None, :] < N), other=0.0)
    out = acc.to(S.dtype.element_ty) + resid
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], out,
             mask=x_mask & (offs_n[None, :] < N))


def fused_mlp_forward(
    X: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    M, K = X.shape
    N = gate_w.shape[0]
    H = down_w.shape[0]

    S = torch.empty((M, N), device=X.device, dtype=X.dtype)
    fused_gateup_swiglu_kernel[(triton.cdiv(M, 64) * triton.cdiv(N, 64),)](
        X, gate_w, up_w, S, M, N, K,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=8,
    )

    Y = torch.empty((M, H), device=X.device, dtype=X.dtype)
    fused_down_residual_kernel[(triton.cdiv(M, 64) * triton.cdiv(H, 64),)](
        S, down_w, residual, Y, M, H, N,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8,
    )
    return Y