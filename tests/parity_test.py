"""Dual-implementation parity check: tiny synthetic Qwen2 vs HF transformers.

Builds a random-weight tiny checkpoint, loads the SAME safetensors into HF
transformers and tinyinfra, and asserts the logits (prefill and one decode
step through the paged KV cache) match. Self-contained: no hub downloads.
"""

import pytest
import torch
from safetensors.torch import save_file

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernels require CUDA")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg_dict = dict(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=128,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=True,
    )
    torch.manual_seed(42)
    hf_config = Qwen2Config(**cfg_dict)
    hf_model = Qwen2ForCausalLM(hf_config).eval()

    ckpt_dir = tmp_path_factory.mktemp("tiny_qwen2")
    import json

    (ckpt_dir / "config.json").write_text(json.dumps(cfg_dict))
    (ckpt_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [151645]}))
    save_file(hf_model.state_dict(), str(ckpt_dir / "model.safetensors"), metadata={"format": "pt"})

    # Move to cuda fp16 for both implementations (matches production)
    hf_model = hf_model.to("cuda", torch.float16)
    return ckpt_dir, hf_model


def _build_engine_cache(config, batch_size=1, max_seq_len=128, device="cuda"):
    from tinyinfra.engine.cache_manager import KVCacheManager

    return KVCacheManager(
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=config.num_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        block_size=16,
        device=torch.device(device),
        dtype=torch.float16,
    )


def _rope_caches(config, device="cuda"):
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(config.max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _engine_logits(ids, model, config, kv_cache, cos_cache, sin_cache):
    batch_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        return model(ids.to("cuda"), batch_indices, kv_cache, cos_cache, sin_cache)


def test_logits_match_prefill(tiny_checkpoint):
    from tinyinfra.engine.loader import load_model

    ckpt_dir, hf_model = tiny_checkpoint
    loaded = load_model(str(ckpt_dir), device="cuda", dtype=torch.float16)
    model, cfg = loaded.model, loaded.config
    kv_cache = _build_engine_cache(cfg)
    cos_cache, sin_cache = _rope_caches(cfg)

    ids = torch.tensor([[7, 123, 256, 42, 511, 100, 7, 88, 300, 1, 5, 12]])
    hf_logits = hf_model(ids.to("cuda")).logits
    ours = _engine_logits(ids, model, cfg, kv_cache, cos_cache, sin_cache)

    torch.testing.assert_close(ours.cpu(), hf_logits.cpu(), atol=0.05, rtol=0.02)
    assert (ours.argmax(-1) == hf_logits.argmax(-1)).all(), "greedy tokens diverged on prefill"


def test_logits_match_after_one_decode_step(tiny_checkpoint):
    from tinyinfra.engine.loader import load_model

    ckpt_dir, hf_model = tiny_checkpoint
    loaded = load_model(str(ckpt_dir), device="cuda", dtype=torch.float16)
    model, cfg = loaded.model, loaded.config
    kv_cache = _build_engine_cache(cfg)
    cos_cache, sin_cache = _rope_caches(cfg)

    prompt = torch.tensor([[7, 123, 256, 42, 511, 100]], device="cuda")
    with torch.no_grad():
        hf_full = hf_model(torch.cat([prompt, torch.tensor([[88]], device="cuda")], dim=1)).logits
    batch_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        _ = model(prompt, batch_indices, kv_cache, cos_cache, sin_cache)
    kv_cache.seq_len[batch_indices] += prompt.shape[1]  # scheduler does this after forward

    next_token = torch.tensor([[88]], device="cuda")
    ours = _engine_logits(next_token, model, cfg, kv_cache, cos_cache, sin_cache)

    torch.testing.assert_close(ours.cpu(), hf_full[:, -1:, :].cpu(), atol=0.05, rtol=0.02)
    assert ours.argmax(-1).item() == hf_full[:, -1].argmax(-1).item(), "greedy tokens diverged after decode step"
