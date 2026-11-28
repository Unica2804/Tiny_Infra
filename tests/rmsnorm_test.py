import pytest
import torch

from tinyinfra.kernels.RMSNorm import fused_add_rmsnorm, rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def ref_norm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


@pytest.mark.parametrize("M,H", [(1, 896), (8, 896), (64, 896), (1, 2048), (32, 2048)])
def test_rmsnorm(M, H):
    torch.manual_seed(M + H)
    x = torch.randn(M, H, device="cuda", dtype=torch.float16)
    w = torch.randn(H, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(rmsnorm(x, w), ref_norm(x, w), atol=0.05, rtol=0.01)


@pytest.mark.parametrize("M,H", [(1, 896), (16, 896), (1, 2048), (32, 2048)])
def test_fused_add_rmsnorm(M, H):
    torch.manual_seed(M + H)
    x = torch.randn(M, H, device="cuda", dtype=torch.float16)
    r = torch.randn(M, H, device="cuda", dtype=torch.float16)
    w = torch.randn(H, device="cuda", dtype=torch.float16)
    y, rout = fused_add_rmsnorm(x, r, w)
    torch.testing.assert_close(rout, x + r, atol=0, rtol=0)
    torch.testing.assert_close(y, ref_norm(x + r, w), atol=0.05, rtol=0.01)
