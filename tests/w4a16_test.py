"""Correctness test for the W4A16 GPTQ (4-bit) kernel against a chunked dequant reference."""

import pytest
import torch

from tinyinfra.kernels.quants.w4a16 import w4a16_matmul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_qweights(K, N, group=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qw = torch.randint(0, 0x7FFFFFFF, (K // 8, N), generator=g,
                       device="cuda", dtype=torch.int32)
    # symmetric zero point 7 in each nibble -> all 0x77777777
    qz = torch.full((K // group, N // 8), 0x77777777, device="cuda", dtype=torch.int32)
    sc = torch.rand(K // group, N, generator=g, device="cuda", dtype=torch.float16) * 0.01 + 0.005
    return qw, qz, sc


def dequant_reference(X, qw, qz, scales, group=128):
    K, N = qw.shape[0] * 8, qw.shape[1]
    acc = torch.zeros((X.shape[0], N), device="cuda", dtype=torch.float32)
    nn = torch.arange(N, device="cuda")
    for k0 in range(0, K, 128):
        kk = torch.arange(k0, min(k0 + 128, K), device="cuda")
        w = ((qw[kk // 8] >> ((kk % 8)[:, None] * 4)) & 0xF).float()
        z = ((qz[kk // group][:, nn // 8] >> ((nn % 8)[None, :] * 4)) & 0xF).float()
        acc += X[:, k0:k0 + len(kk)].float() @ ((w - z) * scales[kk // group].float())
    return acc.to(torch.float16)


@pytest.mark.parametrize("M,K,N", [(1, 896, 4864), (8, 2048, 11008), (64, 2048, 11008)])
def test_w4a16_matches_dequant(M, K, N):
    torch.manual_seed(M)
    X = torch.randn(M, K, device="cuda", dtype=torch.float16)
    qw, qz, sc = make_qweights(K, N)
    y = w4a16_matmul(X, qw, qz, sc, 128)
    ref = dequant_reference(X, qw, qz, sc, 128)
    torch.testing.assert_close(y, ref, atol=0.05, rtol=0.01)
