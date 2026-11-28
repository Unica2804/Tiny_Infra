"""Quantization dispatch: load-time adapters + runtime kernel router."""

from tinyinfra.engine.quant.router import (
    QuantTensor, quant_matmul, register_kernel, resolve_scheme, supported_schemes,
)

_ADAPTERS: dict[tuple[str, int], object] = {}


def register(method: str, bits: int):
    def deco(fn):
        _ADAPTERS[(method, bits)] = fn
        return fn
    return deco


def supported_quant() -> list[tuple[str, int]]:
    return sorted(_ADAPTERS)


def get_adapter(method: str, bits: int):
    try:
        return _ADAPTERS[(method, bits)]
    except KeyError:
        raise ValueError(
            f"No quant adapter for method={method!r}, bits={bits}. "
            f"Supported: {supported_quant()}"
        ) from None


# Import adapters so they register themselves.
from tinyinfra.engine.quant import gptq, compressed_tensors  # noqa: E402,F401
