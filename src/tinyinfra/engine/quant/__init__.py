"""Quantization dispatch: quant_method+bits -> adapter and compute kernel.

Mirrors the model registry in `tinyinfra/models/__init__.py`. Adding a new
checkpoint format or bit-width is a one-line registration here.
"""

from dataclasses import dataclass
from typing import Callable

import torch

_ADAPTERS: dict[tuple[str, int], Callable] = {}


def register(method: str, bits: int):
    """Decorator: register an unpacking adapter for (quant_method, bits)."""
    def deco(fn):
        _ADAPTERS[(method, bits)] = fn
        return fn
    return deco


def supported_quant() -> list[tuple[str, int]]:
    return sorted(_ADAPTERS)


def get_adapter(method: str, bits: int) -> Callable:
    try:
        return _ADAPTERS[(method, bits)]
    except KeyError:
        raise ValueError(
            f"No quant adapter for method={method!r}, bits={bits}. "
            f"Supported: {supported_quant()}"
        ) from None


@dataclass
class QuantTensor:
    """Canonical quantized weight: plain uint8 W [K,N], fp16 scales [K/g,N], uint8 zeros."""
    qweight: torch.Tensor   # uint8 [K, N]
    scales: torch.Tensor    # fp16 [K//group, N]
    zeros: torch.Tensor     # uint8 [K//group, N]
    group: int
    sym: bool
    bias: torch.Tensor | None = None


# Import adapters so they register themselves. Keep at bottom to avoid cycles.
from tinyinfra.engine.quant import gptq, compressed_tensors  # noqa: E402,F401
