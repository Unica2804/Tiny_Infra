import torch
import torch.nn as nn
import custom_swiglu
from engine.config import QwenConfig


class Qwen2MLP(nn.Module):
    def __init__(self, config:QwenConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fused_gate_up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        gate_up = self.fused_gate_up_proj(x)
        hidden_states = custom_swiglu.forward(gate_up)
        return self.down_proj(hidden_states)