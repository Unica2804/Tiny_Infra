"""
    In Place Rotary Positional Embedding (RoPE)
    H_q -> number of query heads
    D -> head dimension
    m-> token index
    h -> head index
    d -> dimension index
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr, pos_ptr,
    stride_qm,stride_qh,stride_qd,
    stride_km,stride_kh,stride_kd,
    stride_cm,stride_cd,
    stride_sm,stride_sd,
    H_q, D,
    BHD : tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    half_D = D // 2
    offs_d = tl.arange(0, BHD)
    mask_d = offs_d < half_D

    pos = tl.load(pos_ptr + pid_m)
    cos = tl.load(cos_ptr + pos * stride_cm + offs_d * stride_cd, mask=mask_d, other=0.0)
    sin = tl.load(sin_ptr + pos * stride_sm + offs_d * stride_sd, mask=mask_d, other=0.0)

    # Route to Q or K based on head index
    if pid_h < H_q:
        base_ptr = q_ptr + pid_m * stride_qm + pid_h * stride_qh
        stride_d = stride_qd
    else:
        base_ptr = k_ptr + pid_m * stride_km + (pid_h - H_q) * stride_kh
        stride_d = stride_kd
    
    left_ptrs = base_ptr + offs_d * stride_d
    right_ptrs = base_ptr + (half_D + offs_d) * stride_d

    x_left = tl.load(left_ptrs, mask=mask_d, other=0.0)
    x_right = tl.load(right_ptrs, mask=mask_d, other=0.0)

    #compute rotated values
    x_left_f32 = tl.cast(x_left, tl.float32)
    x_right_f32 = tl.cast(x_right, tl.float32)

    out_left = (x_left_f32 * cos - x_right_f32 * sin).to(x_left.dtype)
    out_right = (x_left_f32 * sin + x_right_f32 * cos).to(x_right.dtype)

    #Write back to memory
    tl.store(left_ptrs, out_left, mask=mask_d)
    tl.store(right_ptrs, out_right, mask=mask_d)

def rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_ids: torch.Tensor,
):
    """Rotate q and k in place. Same contract as the old CUDA binding."""
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda and pos_ids.is_cuda, (
        "All tensors must be CUDA tensors"
    )
    M, H_q, D = q.shape
    _, H_k, _ = k.shape
    assert q.size(0) == k.size(0) and q.size(2) == k.size(2), "q/k must agree on T and head_dim"
    assert D % 2 == 0, "head_dim must be even"
    assert cos.shape[-1] == D // 2, f"Expected cache width {D//2}, got {cos.shape[-1]}"
    BLOCK_HALF_D = triton.next_power_of_2(D // 2)
    grid = (M, H_q + H_k)

    rope_kernel[grid](
        q, k,
        cos, sin, pos_ids,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        H_q, D,
        BLOCK_HALF_D,
        num_warps=1,
    )