"""Quantized linear layer: drops in for nn.Linear, dispatches to Triton W8A16."""

import torch
import torch.nn as nn

from tinyinfra.engine.quant import QuantTensor
from tinyinfra.kernels.quants.w8a16 import w8a16_matmul


class QuantLinear(nn.Module):
    def __init__(self, qt: QuantTensor):
        super().__init__()
        self.register_buffer("qweight", qt.qweight)
        self.register_buffer("scales", qt.scales)
        self.register_buffer("zeros", qt.zeros)
        self.group = qt.group
        self.sym = qt.sym
        if qt.bias is not None:
            self.register_buffer("bias", qt.bias.clone())
        else:
            self.bias = None
        self.in_features = qt.qweight.shape[0]
        self.out_features = qt.qweight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        out = w8a16_matmul(h, self.qweight, self.scales, self.zeros, self.group)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group={self.group}, sym={self.sym}"
