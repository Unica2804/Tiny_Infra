"""Parity check for compressed-tensors (llm-compressor) W8A16 adapter.

Independent PyTorch dequant (uint8 - 128) * per-channel scale, compared against
the QuantLinear/Triton path on the same engine code.
"""

from pathlib import Path

import pytest
import torch

from tinyinfra.engine.cache_manager import KVCacheManager
from tinyinfra.engine.loader import load_model
from tinyinfra.engine.quant.linear import QuantLinear
from tinyinfra.kernels.quants.w8a16 import w8a16_matmul

MODEL = "weights/qwenw8a16"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (Path(MODEL) / "model.safetensors").is_file(),
    reason="requires CUDA and a local Qwen2.5-W8A16 snapshot in weights/",
)


def test_compressed_tensors_w8a16_adapter():
    loaded = load_model(MODEL, device="cuda", dtype=torch.float16)
    model = loaded.model
    assert sum(isinstance(m, QuantLinear) for m in model.modules()) == 168

    # Verify a single layer's kernel output against independent dequant math.
    ql = model.model.layers[0].self_attn.q_proj
    assert isinstance(ql, QuantLinear)
    x = torch.randn(16, ql.in_features, device="cuda", dtype=torch.float16)
    y = w8a16_matmul(x, ql.qweight, ql.scales, ql.zeros, ql.group)
    ref = (x.float() @ (ql.qweight.float() - ql.zeros.float()) * ql.scales.float()).to(torch.float16)
    torch.testing.assert_close(y, ref, atol=0.05, rtol=0.01)

    # End-to-end forward finite.
    c = loaded.config
    hd = c.hidden_size // c.num_attention_heads
    kv = KVCacheManager(1, 128, c.num_layers, c.num_key_value_heads, hd,
                        16, torch.device("cuda"), torch.float16)
    inv = 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    fr = torch.outer(torch.arange(128, device="cuda", dtype=torch.float32), inv)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda", dtype=torch.int32)
    bi = torch.zeros(1, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        out = model(ids, bi, kv, fr.cos().contiguous(), fr.sin().contiguous())
    assert torch.isfinite(out).all()
