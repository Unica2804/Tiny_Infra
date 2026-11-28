import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['H', 'ADD'],
)
@triton.jit
def rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
    H, eps,
    ADD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per row
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H

    # load input, optionally add residual
    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0)
    if ADD:
        r = tl.load(res_ptr + row * H + offs, mask=mask, other=0.0)
        x = x + r

    # variance in fp32
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, 0) / H

    # normalize and scale
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xf * tl.rsqrt(var + eps) * w

    # store output and optional summed residual
    tl.store(out_ptr + row * H + offs, y.to(tl.float16), mask=mask)
    if ADD:
        tl.store(res_out_ptr + row * H + offs, x.to(tl.float16), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    h = x.reshape(-1, shape[-1])
    out = torch.empty_like(h)
    rmsnorm_kernel[(h.shape[0],)](
        h, h, weight, out, h, h.shape[1], eps, False,
        BLOCK=triton.next_power_of_2(h.shape[1]),
    )
    return out.view(shape)


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    h = x.reshape(-1, shape[-1])
    r = residual.reshape(-1, shape[-1])
    out = torch.empty_like(h)
    res_out = torch.empty_like(h)
    rmsnorm_kernel[(h.shape[0],)](
        h, r, weight, out, res_out, h.shape[1], eps, True,
        BLOCK=triton.next_power_of_2(h.shape[1]),
    )
    return out.view(shape), res_out.view(shape)
