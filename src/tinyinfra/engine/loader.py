"""General-purpose checkpoint loading — native HF `.safetensors`, no extraction.

Handles any architecture registered in `tinyinfra.models` (currently `qwen2`).
A `model_ref` is either a local snapshot directory or a Hugging Face hub id
(e.g. "Qwen/Qwen2.5-0.5B-Instruct"), which is downloaded via `huggingface_hub`.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from tinyinfra.engine.quant import get_adapter
from tinyinfra.engine.quant.linear import QuantLinear
from tinyinfra.models import (
    get_config_class,
    get_model_class,
    supported_model_types,
)

logger = logging.getLogger("tinyinfra.loader")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    config: object  # QwenConfig today; per-arch config classes as they land
    model_dir: Path


def resolve_model_dir(model_ref: str | os.PathLike) -> Path:
    """Resolve a hub id or local directory to a concrete snapshot directory."""
    if os.path.isdir(model_ref):
        return Path(model_ref)
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s (weights + config/tokenizer metadata)...", model_ref)
    return Path(snapshot_download(str(model_ref), allow_patterns=["*.safetensors", "*.json"]))


def load_eos_token_ids(model_dir: str | Path) -> list[int]:
    """EOS ids from generation_config.json — it may be a single int or a list."""
    gc_path = Path(model_dir) / "generation_config.json"
    if not gc_path.is_file():
        return []
    eos = json.loads(gc_path.read_text()).get("eos_token_id")
    if eos is None:
        return []
    return [int(e) for e in eos] if isinstance(eos, list) else [int(eos)]


def load_model(
    model_ref: str | os.PathLike,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Load a native checkpoint. No custom weight layouts, no translation step."""
    model_dir = resolve_model_dir(model_ref)
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.json at {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    model_type = cfg.get("model_type")
    try:
        config = get_config_class(model_type).from_dict(cfg)
        model_cls = get_model_class(model_type)
    except ValueError as e:
        raise ValueError(
            f"{e}\nArchitectures in checkpoint: {cfg.get('architectures')} | config: {cfg_path}"
        ) from e

    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = model_cls(config)
    expected = set(model.state_dict().keys())
    quant_cfg = cfg.get("quantization_config")
    quant_tensors: dict[str, dict] = {}       # module path -> {qweight,qzeros,scales,g_idx,bias?}
    if quant_cfg:
        method, bits = quant_cfg["quant_method"], int(quant_cfg.get("bits", 4))
        group = int(quant_cfg.get("group_size", 128))
        sym = bool(quant_cfg.get("sym", True))
        skip = set(quant_cfg.get("modules_to_not_convert") or [])

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found under {model_dir}")

    loaded, unexpected = set(), set()
    for shard in shards:
        logger.info("Loading %s", shard.name)
        state = load_file(str(shard))
        if quant_cfg:
            for key in list(state):
                if ".qweight" in key or ".qzeros" in key or ".scales" in key or ".g_idx" in key:
                    mod, field = key.rsplit(".", 1)
                    quant_tensors.setdefault(mod, {})[field] = state.pop(key)
        _, unexp = model.load_state_dict(state, strict=False)
        loaded |= set(state.keys())
        unexpected |= set(unexp)
        del state

    if quant_cfg:
        adapter = get_adapter(method, bits)
        logger.info("Quantized checkpoint: %s bits=%d group=%d -> %d layers",
                    method, bits, group, len(quant_tensors))
        for mod_path, tensors in quant_tensors.items():
            if "qweight" not in tensors:
                continue
            parent_path, _, name = mod_path.rpartition(".")
            parent = model.get_submodule(parent_path)
            linear = model.get_submodule(mod_path)
            bias = getattr(linear, "bias", None)
            # g_idx maps each K row to a quant group. Kernel assumes sequential
            # grouping (g_idx[i] == i // group), which is what desc_act=false produces.
            gi = tensors.get("g_idx")
            if gi is not None and not torch.equal(gi.long(), torch.arange(len(gi), device=gi.device) // group):
                raise NotImplementedError(f"act-order (non-sequential g_idx) at {mod_path} not supported")
            qt = adapter(tensors["qweight"], tensors["qzeros"], tensors["scales"],
                         group, sym, bias=bias)
            setattr(parent, name, QuantLinear(qt))
        for mod_path in quant_tensors:
            unexpected.discard(mod_path + ".qweight")
            unexpected.discard(mod_path + ".qzeros")
            unexpected.discard(mod_path + ".scales")
            unexpected.discard(mod_path + ".g_idx")

    missing = expected - loaded
    if quant_cfg:
        # replaced nn.Linear.weight keys no longer exist on the model
        for mod_path in quant_tensors:
            missing.discard(mod_path + ".weight")
    # Tied checkpoints omit lm_head.weight — alias it to the embedding matrix.
    if config.tie_word_embeddings:
        missing.discard("lm_head.weight")

    if missing or unexpected:
        msg = [
            f"Checkpoint does not match {model_cls.__name__} (from {cfg_path}).",
        ]
        if missing:
            msg.append(f"Missing keys ({len(missing)}): {sorted(missing)[:20]}")
        if unexpected:
            msg.append(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:20]}")
        raise RuntimeError("\n".join(msg))

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
        logger.info("Tied lm_head.weight -> embed_tokens.weight")

    model = model.to(device=device, dtype=dtype).eval()
    logger.info(
        "Loaded %s (%s): %d layers, vocab %d -> %s",
        model_ref,
        model_type,
        config.num_layers,
        config.vocab_size,
        device,
    )
    return LoadedModel(model=model, config=config, model_dir=model_dir)


if __name__ == "__main__":
    import sys

    ref = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    out = load_model(ref)
    print(f"OK: {ref} -> {out.config.num_layers} layers, dir {out.model_dir}")
