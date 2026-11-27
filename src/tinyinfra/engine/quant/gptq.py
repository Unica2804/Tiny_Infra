"""GPTQ W8A16 (bits=8) adapter: packed int32 qweight/qzeros -> canonical uint8."""

import torch

from tinyinfra.engine.quant import QuantTensor, register


def _unpack4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack 4 int8 values per int32 along dim 0: [K/4, N] -> [K, N] uint8."""
    N = packed.shape[1]
    out = torch.empty((K, N), dtype=torch.uint8, device=packed.device)
    for j in range(4):
        out[j::4] = ((packed >> (8 * j)) & 0xFF).to(torch.uint8)
    return out


@register("gptq", 8)
def unpack_gptq8(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
                 group: int, sym: bool, bias=None) -> QuantTensor:
    K_packed, N = qweight.shape
    K = K_packed * 4
    W = _unpack4(qweight, K)                                   # uint8 [K,N]
    # qzeros packs 4 int8 per int32 along dim 1: [K/g, N/4] -> [K/g, N]
    Z = torch.empty((qzeros.shape[0], N), dtype=torch.uint8, device=qzeros.device)
    for j in range(4):
        Z[:, j::4] = ((qzeros >> (8 * j)) & 0xFF).to(torch.uint8)
    return QuantTensor(qweight=W, scales=scales.to(torch.float16),
                       zeros=Z.to(torch.uint8), group=group, sym=sym, bias=bias)
