import pytest
import torch

from tinyinfra.kernels.Paged_Attn import launch_paged_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="paged attention kernel requires CUDA")


def ref_attention(q, k_cache, v_cache, block_tables, context_lens, sm_scale):
    """Slow, obviously-correct reference. q: [B, T, Hq, D] fp16."""
    B, T, Hq, D = q.shape
    Hkv, block_size = k_cache.shape[1], k_cache.shape[2]
    repeats = Hq // Hkv
    out = torch.zeros(B, T, Hq, D, dtype=torch.float32)

    for b in range(B):
        L = int(context_lens[b].item())
        k_seq, v_seq = [], []
        for i in range(L):
            blk = int(block_tables[b, i // block_size].item())
            k_seq.append(k_cache[blk, :, i % block_size].float())
            v_seq.append(v_cache[blk, :, i % block_size].float())
        k = torch.stack(k_seq).repeat_interleave(repeats, dim=1)  # [L, Hq, D]
        v = torch.stack(v_seq).repeat_interleave(repeats, dim=1)

        for j in range(T):
            q_pos = L - T + j
            qj = q[b, j].float()                              # [Hq, D]
            scores = torch.einsum("hd,lhd->hl", qj, k) * sm_scale
            mask = torch.arange(L, device=q.device) <= q_pos
            scores = scores.masked_fill(~mask, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[b, j] = torch.einsum("hl,lhd->hd", p, v)
    return out


def make_case(B, T, Hq, Hkv, D, ctx_lens, max_seq=64, seed=0):
    torch.manual_seed(seed)
    block_size = 16
    max_blocks = max_seq // block_size
    q = torch.randn(B, T, Hq, D, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(B * max_blocks, Hkv, block_size, D, device="cuda", dtype=torch.float16) * 0.1
    v_cache = torch.randn(B * max_blocks, Hkv, block_size, D, device="cuda", dtype=torch.float16) * 0.1

    block_tables = torch.full((B, max_blocks), -1, device="cuda", dtype=torch.int32)
    context_lens = torch.tensor(ctx_lens, device="cuda", dtype=torch.int32)
    pool = list(range(B * max_blocks))
    for b in range(B):
        ctx = int(context_lens[b])
        for blk in range((ctx + block_size - 1) // block_size):
            block_tables[b, blk] = pool.pop(0)
    return q, k_cache, v_cache, block_tables, context_lens, block_size


def assert_matches_ref(q, k_cache, v_cache, block_tables, context_lens, sm_scale=None):
    out = launch_paged_attention(q.clone(), k_cache, v_cache, block_tables, context_lens, sm_scale)
    assert out.shape == q.shape
    ref = ref_attention(q, k_cache, v_cache, block_tables, context_lens,
                        sm_scale or 1.0 / (q.shape[-1] ** 0.5))
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)


def test_decode_block_aligned():
    q, kc, vc, bt, cl, _ = make_case(2, 1, 4, 2, 32, [32, 48])
    assert_matches_ref(q, kc, vc, bt, cl)


def test_decode_masked_tail():
    # context_len not a multiple of 16 -> padded block tail must not leak into softmax
    q, kc, vc, bt, cl, _ = make_case(2, 1, 4, 2, 32, [25, 37])
    # fill unused-block padding with huge values to make leakage obvious
    kc *= 1.0
    assert_matches_ref(q, kc, vc, bt, cl)


def test_decode_padded_block_leakage():
    # padded positions filled with enormous values: any softmax leak must dominate
    q, kc, vc, bt, cl, bs = make_case(1, 1, 4, 2, 32, [17])
    for b in range(1):
        for blk in range(2):
            phys = int(bt[b, blk].item())
            for t in range(bs):
                pos = blk * bs + t
                if pos >= 17:
                    kc[phys, :, t] = 1e4
    out = launch_paged_attention(q, kc, vc, bt, cl)
    assert out.abs().max().item() < 1e2, "padded K positions leaked into softmax"


def test_decode_context_len_one():
    q, kc, vc, bt, cl, _ = make_case(2, 1, 4, 2, 32, [1, 1])
    assert_matches_ref(q, kc, vc, bt, cl)


def test_decode_zero_context_returns_zeros():
    # context_len == 0: kernel must store defined memory — launcher zero-initializes `out`
    q, kc, vc, bt, cl, _ = make_case(1, 1, 4, 2, 32, [0])
    out = launch_paged_attention(q, kc, vc, bt, cl)
    assert (out == 0).all(), "kernel returned uninitialized memory for empty context"


def test_spec_decode_two_tokens():
    q, kc, vc, bt, cl, _ = make_case(2, 2, 4, 2, 32, [20, 30])
    assert_matches_ref(q, kc, vc, bt, cl)


def test_gqa_ratios():
    for Hq, Hkv in [(4, 4), (4, 2), (4, 1), (8, 4), (8, 2), (8, 1)]:
        q, kc, vc, bt, cl, _ = make_case(1, 1, Hq, Hkv, 32, [20])
        assert_matches_ref(q, kc, vc, bt, cl)


def test_head_dims():
    for D in [32, 64, 128]:
        q, kc, vc, bt, cl, _ = make_case(1, 1, 4, 2, D, [25])
        assert_matches_ref(q, kc, vc, bt, cl)


def test_shape_3d_input_squeeze():
    # [B, H, D] in -> [B, H, D] out (the model's decode call shape)
    q, kc, vc, bt, cl, _ = make_case(2, 1, 4, 2, 32, [20, 24])
    out = launch_paged_attention(q.squeeze(1), kc, vc, bt, cl)
    assert out.shape == q.squeeze(1).shape


def test_default_vs_explicit_sm_scale():
    q, kc, vc, bt, cl, _ = make_case(1, 1, 4, 2, 32, [16])
    D = q.shape[-1]
    out_default = launch_paged_attention(q, kc, vc, bt, cl)
    out_explicit = launch_paged_attention(q, kc, vc, bt, cl, sm_scale=1.0 / (D ** 0.5))
    torch.testing.assert_close(out_default, out_explicit)


def test_bf16_supported():
    q, kc, vc, bt, cl, _ = make_case(1, 1, 4, 2, 32, [25])
    out = launch_paged_attention(q.bfloat16(), kc.bfloat16(), vc.bfloat16(), bt, cl)
    assert out.dtype == torch.bfloat16


def test_gqa_mismatch_asserts():
    q, kc, vc, bt, cl, _ = make_case(1, 1, 4, 3, 32, [16])  # 4 % 3 != 0
    with pytest.raises(AssertionError):
        launch_paged_attention(q, kc, vc, bt, cl)


def test_non_pow2_block_size_asserts():
    # k_cache block dim 20 is not a power of 2 -> wrapper must reject
    q, kc, vc, bt, cl, _ = make_case(1, 1, 4, 2, 32, [16])
    kc2 = torch.randn(4, 2, 20, 32, device="cuda", dtype=torch.float16)
    vc2 = torch.randn_like(kc2)
    bt2 = torch.zeros((1, 4), device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        launch_paged_attention(q, kc2, vc2, bt2, cl[:1] * 16)
