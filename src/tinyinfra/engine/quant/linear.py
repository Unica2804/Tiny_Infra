"""Quantized linear layer: nn.Linear drop-in, dispatches via the quant router."""

import torch
import torch.nn as nn

from tinyinfra.engine.quant.router import QuantTensor, quant_matmul


class QuantLinear(nn.Module):
    def __init__(self, qt: QuantTensor):
        super().__init__()
        self.scheme = qt.scheme
        for name in ("qweight", "scales", "zeros", "qzeros"):
            t = getattr(qt, name, None)
            if t is not None:
                self.register_buffer(name, t)
        self.group = qt.group
        self.sym = qt.sym
        if qt.bias is not None:
            self.register_buffer("bias", qt.bias.clone())
        else:
            self.bias = None
        self.in_features = qt.qweight.shape[0] * (8 if self.scheme == "W4A16" else 1)
        self.out_features = qt.qweight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = quant_matmul(self.scheme, x.reshape(-1, shape[-1]), self)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"scheme={self.scheme}, in={self.in_features}, out={self.out_features}, group={self.group}"
