import pytest
import torch

from tinyinfra.kernels.Rope import rope_inplace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE kernel requires CUDA")


def ref_rope(q, k, cos, sin, pos):
    cos = cos[pos].unsqueeze(1)  # [T, 1, D//2] -> broadcast over heads
    sin = sin[pos].unsqueeze(1)

    def rotate(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half].float(), x[..., half:].float()
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def _make_case(T, num_q_heads, num_kv_heads, head_dim, max_seq=32, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(T, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(max_seq, head_dim // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn(max_seq, head_dim // 2, device="cuda", dtype=torch.float32)
    pos = torch.randint(0, max_seq, (T,), device="cuda", dtype=torch.int32)
    return q, k, cos, sin, pos


def _check_matches_reference(q, k, cos, sin, pos):
    q_ref, k_ref = ref_rope(q.clone(), k.clone(), cos, sin, pos)
    rope_inplace(q, k, cos, sin, pos)  # rotates in place
    torch.testing.assert_close(q, q_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, k_ref, atol=1e-3, rtol=1e-3)


def test_rope():
    q, k, cos, sin, pos = _make_case(B := 2, num_q_heads=4, num_kv_heads=4, head_dim=8)
    _check_matches_reference(q, k, cos, sin, pos)


def test_rope_gqa_asymmetry():
    # Real Qwen2.5-0.5B geometry: 14 q heads, 2 kv heads, head_dim 64
    q, k, cos, sin, pos = _make_case(5, num_q_heads=14, num_kv_heads=2, head_dim=64)
    _check_matches_reference(q, k, cos, sin, pos)


def test_rope_decode_single_token():
    q, k, cos, sin, pos = _make_case(1, num_q_heads=14, num_kv_heads=2, head_dim=64)
    _check_matches_reference(q, k, cos, sin, pos)


def test_rope_masked_non_pow2_half():
    # half_dim = 12, next_pow2 = 16 -> exercises the offs_d mask
    q, k, cos, sin, pos = _make_case(3, num_q_heads=2, num_kv_heads=1, head_dim=24, max_seq=16)
    _check_matches_reference(q, k, cos, sin, pos)


def test_rope_non_contiguous_views():
    # The kernel reads strides, so views that aren't row-contiguous must still rotate correctly.
    # Build from a transposed buffer: logical [T, H, D] with H-major physical layout.
    q, k, cos, sin, pos = _make_case(4, num_q_heads=4, num_kv_heads=2, head_dim=16)
    q_t = q.transpose(0, 1).contiguous().transpose(0, 1)  # physically [H, T, D]
    k_t = k.transpose(0, 1).contiguous().transpose(0, 1)
    assert not q_t.is_contiguous() and q_t.shape == q.shape
    _check_matches_reference(q_t, k_t, cos, sin, pos)


def test_rope_edge_positions():
    # pos at the very beginning and very end of the table
    cos = torch.randn(4, 4, device="cuda", dtype=torch.float32)
    sin = torch.randn(4, 4, device="cuda", dtype=torch.float32)
    q = torch.randn(2, 2, 8, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, 8, device="cuda", dtype=torch.float16)
    pos = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    _check_matches_reference(q, k, cos, sin, pos)
