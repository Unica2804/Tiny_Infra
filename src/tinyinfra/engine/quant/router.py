"""Quant router: checkpoint config -> scheme, scheme -> kernel.

Two dispatch points:
  load time  - adapters unpack checkpoint bytes into canonical tensors
  runtime    - this router picks the GEMM kernel for each quantized linear layer
"""

from dataclasses import dataclass
from typing import Callable

import torch

_KERNELS: dict[str, Callable] = {}


def register_kernel(scheme: str):
    def deco(fn):
        _KERNELS[scheme] = fn
        return fn
    return deco


def supported_schemes() -> list[str]:
    return sorted(_KERNELS)


def quant_matmul(scheme: str, x: torch.Tensor, qt: "QuantTensor") -> torch.Tensor:
    try:
        return _KERNELS[scheme](x, qt)
    except KeyError:
        raise ValueError(f"No kernel for scheme {scheme!r}; supported: {supported_schemes()}") from None


def resolve_scheme(qcfg: dict) -> str:
    method = qcfg["quant_method"]
    bits = qcfg.get("bits")
    act = None
    if bits is None:  # compressed-tensors
        cg = qcfg.get("config_groups") or {}
        w = next(iter(cg.values())).get("weights", {}) if cg else {}
        bits = w.get("num_bits", w.get("bits", 4))
        act = w.get("input_activations")
    bits = int(bits)
    if act is not None:
        abits = int(act.get("num_bits", act.get("bits", 0)))
        return f"W{bits}A{abits}"
    return f"W{bits}A16"


@dataclass
class QuantTensor:
    scheme: str
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None = None
    qzeros: torch.Tensor | None = None
    group: int = 128
    sym: bool = True
    bias: torch.Tensor | None = None


# Register kernel adapters. Imported after definitions to avoid cycles.
from tinyinfra.engine.quant import kernels  # noqa: E402,F401
