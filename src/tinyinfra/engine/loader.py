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

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found under {model_dir}")

    loaded, unexpected = set(), set()
    for shard in shards:
        logger.info("Loading %s", shard.name)
        state = load_file(str(shard))
        _, unexp = model.load_state_dict(state, strict=False)
        loaded |= set(state.keys())
        unexpected |= set(unexp)
        del state

    missing = expected - loaded
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
