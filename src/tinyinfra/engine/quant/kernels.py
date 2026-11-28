import torch

"""Kernel adapters for the quant router — uniform fn(x, qt) -> y signature."""

from tinyinfra.engine.quant.router import QuantTensor, register_kernel
from tinyinfra.kernels.quants.w4a16 import w4a16_matmul
from tinyinfra.kernels.quants.w8a16 import w8a16_matmul
from tinyinfra.kernels.quants.w8a8 import quantize_int8, w8a8_matmul


@register_kernel("W8A16")
def _w8a16(x: torch.Tensor, qt: QuantTensor) -> torch.Tensor:
    return w8a16_matmul(x, qt.qweight, qt.scales, qt.zeros, qt.group)


@register_kernel("W4A16")
def _w4a16(x: torch.Tensor, qt: QuantTensor) -> torch.Tensor:
    return w4a16_matmul(x, qt.qweight, qt.qzeros, qt.scales, qt.group)


@register_kernel("W8A8")
def _w8a8(x: torch.Tensor, qt: QuantTensor) -> torch.Tensor:
    x_q, x_scales = quantize_int8(x)
    return w8a8_matmul(x_q, qt.qweight, x_scales, qt.scales, qt.group)
