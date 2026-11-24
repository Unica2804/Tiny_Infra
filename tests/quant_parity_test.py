"""Independent parity check for GPTQ-Int8 W8A16 hooks.

Builds a dequantized fp16 reference model directly from the raw checkpoint
tensors (plain PyTorch unpack + dequant) and compares logits against the
QuantLinear/Triton path. No `optimum`/`gptqmodel` dependency needed.
"""

from pathlib import Path

import pytest
import torch

from tinyinfra.engine.cache_manager import KVCacheManager
from tinyinfra.engine.loader import load_model
from tinyinfra.engine.quant.linear import QuantLinear
from tinyinfra.models import get_model_class

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"


def _unpack4(packed: torch.Tensor, rows: int) -> torch.Tensor:
    out = torch.empty((rows, packed.shape[1]), dtype=torch.uint8, device=packed.device)
    for j in range(4):
        out[j::4] = ((packed >> (8 * j)) & 0xFF).to(torch.uint8)
    return out


def _build_dequant_state(ckpt_dir: Path, cfg: dict, device: str = "cuda"):
    """Create a standard fp16 state_dict by dequantizing raw GPTQ-Int8 tensors."""
    from safetensors import safe_open

    group = int(cfg["quantization_config"]["group_size"])
    state = {}
    with safe_open(str(ckpt_dir / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.endswith(".qweight"):
                mod = k[:-len(".qweight")]
                qw = f.get_tensor(k)
                qz = f.get_tensor(mod + ".qzeros")
                sc = f.get_tensor(mod + ".scales")
                W = _unpack4(qw, qw.shape[0] * 4).float()       # [K,N]
                Z = torch.empty((qz.shape[0], qw.shape[1]), dtype=torch.uint8, device=qz.device)
                for j in range(4):
                    Z[:, j::4] = ((qz >> (8 * j)) & 0xFF).to(torch.uint8)
                Z = Z.float()                                    # [K/g,N]
                K, N = W.shape
                wf = torch.empty_like(W)
                for gi in range(K // group):
                    wf[gi*group:(gi+1)*group] = (
                        W[gi*group:(gi+1)*group] - Z[gi][None, :]
                    ) * sc[gi].float()[None, :]
                state[mod + ".weight"] = wf.T.to(torch.float16).contiguous()  # [N,K]
                bias_key = mod + ".bias"
                if bias_key in f.keys():
                    state[bias_key] = f.get_tensor(bias_key).to(torch.float16)
            elif not any(k.endswith(s) for s in (".qzeros", ".scales", ".g_idx")):
                t = f.get_tensor(k)
                state[k] = t.to(torch.float16) if t.is_floating_point() else t
    return state


def test_gptq_int8_matches_dequant_reference():
    from huggingface_hub import snapshot_download

    ckpt = Path(snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"]))
    loaded = load_model(MODEL, device="cuda", dtype=torch.float16)
    model = loaded.model
    assert sum(isinstance(m, QuantLinear) for m in model.modules()) == 168

    import json
    cfg = json.loads((ckpt / "config.json").read_text())
    ref_state = _build_dequant_state(ckpt, cfg)
    ref_model = get_model_class("qwen2")(loaded.config).to("cuda", torch.float16).eval()
    ref_model.load_state_dict(ref_state, strict=False)
    if loaded.config.tie_word_embeddings:
        ref_model.lm_head.weight = ref_model.model.embed_tokens.weight

    c = loaded.config
    hd = c.hidden_size // c.num_attention_heads
    kv = KVCacheManager(1, 128, c.num_layers, c.num_key_value_heads, hd,
                        16, torch.device("cuda"), torch.float16)
    kv_ref = KVCacheManager(1, 128, c.num_layers, c.num_key_value_heads, hd,
                            16, torch.device("cuda"), torch.float16)
    inv = 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    freqs = torch.outer(torch.arange(128, device="cuda", dtype=torch.float32), inv)
    cos, sin = freqs.cos().contiguous(), freqs.sin().contiguous()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda", dtype=torch.int32)
    bi = torch.zeros(1, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        ours = model(ids, bi, kv, cos, sin)
        ref = ref_model(ids, bi, kv_ref, cos, sin)
    assert torch.allclose(ours.float(), ref.float(), atol=0.25, rtol=0.02), (
        f"max diff {(ours.float()-ref.float()).abs().max().item():.3f}"
    )
