import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in (1, 2, 4)
        for s in (2, 3, 4)
    ],
    # One tuned config per (head geometry, kv block size, q_len) — stable per model.
    key=["HEAD_DIM", "BLOCK_SIZE", "Q_LENGTH", "NUM_Q_HEADS", "NUM_KV_HEADS"],
)
@triton.jit
def paged_attention_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    block_table_ptr, context_length_ptr,
    out_ptr,sm_scale,
    stride_qb,stride_qq,stride_qh,stride_qd,
    stride_kb,stride_kh,stride_kt,stride_kd,
    stride_vb,stride_vh,stride_vt,stride_vd,
    stride_bt_b,stride_bt_s,
    stride_ob,stride_oq,stride_oh,stride_od,
    HEAD_DIM : tl.constexpr,
    BD: tl.constexpr,
    NUM_Q_HEADS : tl.constexpr,
    NUM_KV_HEADS : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
    Q_LENGTH : tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    context_len = tl.load(context_length_ptr + seq_idx)
    if context_len <= 0:
        return
    
    group_size : tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    offs_d = tl.arange(0, BD)
    mask_d = offs_d < HEAD_DIM
    offs_t = tl.arange(0, BLOCK_SIZE)

    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)
    bt_row_ptr = block_table_ptr + seq_idx * stride_bt_b

    k_head_base = k_cache_ptr + kv_head_idx * stride_kh
    v_head_base = v_cache_ptr + kv_head_idx * stride_vh

    if Q_LENGTH == 1:
        offs_q = seq_idx * stride_qb + q_head_idx * stride_qh
        q_tile = tl.load(q_ptr + offs_q + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
        q_tile = q_tile * sm_scale

        m_i = float('-inf')
        d_i = 0.0
        acc = tl.zeros([BD], dtype=tl.float32)

        for b in range(0, num_blocks):
            phys_block_idx = tl.load(bt_row_ptr + b * stride_bt_s)
            if phys_block_idx >= 0:
                token_idx = b * BLOCK_SIZE + offs_t
                mask_t = token_idx < context_len
                kv_mask = mask_t[:,None] & mask_d[None,:]

                k_ptrs = (
                    k_head_base 
                    + phys_block_idx * stride_kb 
                    + offs_t[:,None] * stride_kt 
                    + offs_d[None,:] * stride_kd
                )
                k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

                scores = tl.sum(q_tile[None,:] * k_tile, axis=1)
                scores = tl.where(mask_t, scores, float('-inf'))

                # Future extensions here ->

                m_curr = tl.max(scores, axis=0)
                m_new = tl.maximum(m_i, m_curr)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)
                
                d_i = d_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha

                v_ptrs = (
                    v_head_base
                    + phys_block_idx * stride_vb
                    + offs_t[:,None] * stride_vt
                    + offs_d[None,:] * stride_vd
                )
                v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
                acc += tl.sum(p[:,None] * v_tile, axis=0)
                m_i = m_new

        out = acc / d_i
        out_offs = seq_idx * stride_ob + q_head_idx * stride_oh
        tl.store(out_ptr + out_offs + offs_d * stride_od, out.to(out_ptr.dtype.element_ty), mask=mask_d)
    
    else:
        # Speculative Path
        offs_q = tl.arange(0, Q_LENGTH)
        q_ptrs = (
            q_ptr
            + seq_idx * stride_qb
            + offs_q[:,None] * stride_qq
            + q_head_idx * stride_qh
            + offs_d[None,:] * stride_qd
        )
        q_tile = tl.load(q_ptrs, mask=mask_d[None,:], other=0.0).to(tl.float32)
        m_i = tl.full([Q_LENGTH], -float('inf'), dtype=tl.float32)
        d_i = tl.zeros([Q_LENGTH], dtype=tl.float32)
        acc = tl.zeros([Q_LENGTH, BD], dtype=tl.float32)

        q_pos = (context_len - Q_LENGTH) + offs_q

        for b in range(0, num_blocks):
            phys_block_idx = tl.load(bt_row_ptr + b * stride_bt_s)
            if phys_block_idx >= 0:
                token_idx = b * BLOCK_SIZE + offs_t
                mask_t = token_idx < context_len
                kv_mask = mask_t[:,None] & mask_d[None,:]

                k_ptrs = (
                    k_head_base 
                    + phys_block_idx * stride_kb 
                    + offs_t[:,None] * stride_kt 
                    + offs_d[None,:] * stride_kd
                )
                k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

                scores = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
                causal_mask = (token_idx[None,:] <= q_pos[:,None]) & mask_t[None,:]
                scores = tl.where(causal_mask, scores, float('-inf'))

                m_curr = tl.max(scores, axis=1)
                m_new = tl.maximum(m_i, m_curr)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new[:,None])

                d_i = d_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:,None]

                v_ptrs = (
                    v_head_base
                    + phys_block_idx * stride_vb
                    + offs_t[:,None] * stride_vt
                    + offs_d[None,:] * stride_vd
                )
                v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)
                acc = tl.dot(p.to(v_ptrs.dtype.element_ty), v_tile,acc)
                m_i = m_new

        out = acc / d_i[:,None]
        out_ptrs = (
            out_ptr
            + seq_idx * stride_ob
            + offs_q[:,None] * stride_oq
            + q_head_idx * stride_oh
            + offs_d[None,:] * stride_od
        )
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_d[None,:])


def launch_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_length: torch.Tensor,
    sm_scale: float|None = None,
)-> torch.Tensor :
    squeeze_q = q.ndim == 3
    if squeeze_q:
        q = q.unsqueeze(1)

    # Contract validation — fail with a clear message instead of a Triton stack trace.
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda \
        and block_tables.is_cuda and context_length.is_cuda, "all tensors must be CUDA"
    assert q.dtype == k_cache.dtype == v_cache.dtype, (
        f"q/k_cache/v_cache dtype mismatch: {q.dtype} vs {k_cache.dtype} vs {v_cache.dtype}"
    )
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32), f"unsupported dtype {q.dtype}"
    assert block_tables.dtype in (torch.int32, torch.int64), "block_tables must be int32/int64"
    assert context_length.dtype in (torch.int32, torch.int64), "context_length must be int32/int64"
    assert k_cache.shape == v_cache.shape, "k_cache and v_cache must have the same shape"

    batch_size, q_len, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, head_dim_cache = k_cache.shape
    assert head_dim_cache == head_dim, (
        f"head_dim mismatch: q={head_dim}, k_cache={head_dim_cache}"
    )
    # zeros_like: context_len<=0 rows return without writing out — caller must see defined memory
    out = torch.zeros_like(q)

    assert num_q_heads % num_kv_heads == 0, "Number of query heads must be divisible by number of key/value heads"
    assert block_size & (block_size - 1) == 0, "Block size must be a power of 2"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    grid = (num_q_heads, batch_size)

    paged_attention_kernel[grid](
        q, k_cache, v_cache,
        block_tables, context_length,
        out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        HEAD_DIM=head_dim, BD=BLOCK_D, NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads, BLOCK_SIZE=block_size, Q_LENGTH=q_len,
    )

    return out.squeeze(1) if squeeze_q else out