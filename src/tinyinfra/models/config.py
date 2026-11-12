"""Model configuration, parsed from a checkpoint's `config.json`.

No values are invented here — every field comes from the file shipped with the
checkpoint. Defaults exist only where the field is genuinely optional in the
HF format (e.g. `rope_theta` in older Qwen2 configs).
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QwenConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 32768
    head_dim: int | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, cfg: dict) -> "QwenConfig":
        required = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",  # HF calls it num_hidden_layers
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(
                f"config.json is missing required key(s): {', '.join(missing)}"
            )
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            intermediate_size=int(cfg["intermediate_size"]),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 32768)),
            head_dim=cfg.get("head_dim"),  # Some configs state it explicitly (e.g. Qwen3)
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "QwenConfig":
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"No config.json at {cfg_path}")
        return cls.from_dict(json.loads(cfg_path.read_text()))
