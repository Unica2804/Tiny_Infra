"""Model registry: HF `model_type` -> (config class, model class).

To add a new architecture, implement `tinyinfra/models/<arch>.py` and register
it here — one line. The checkpoint loader (`tinyinfra.engine.loader`) dispatches
through this table, so it stays architecture-agnostic.
"""

from tinyinfra.models.config import QwenConfig
from tinyinfra.models.qwen_2 import Qwen2ForCausalLM

_REGISTRY = {
    "qwen2": (QwenConfig, Qwen2ForCausalLM),
}


def supported_model_types() -> list[str]:
    return sorted(_REGISTRY)


def get_config_class(model_type: str):
    try:
        return _REGISTRY[model_type][0]
    except KeyError:
        raise _unsupported(model_type) from None


def get_model_class(model_type: str):
    try:
        return _REGISTRY[model_type][1]
    except KeyError:
        raise _unsupported(model_type) from None


def _unsupported(model_type: str) -> ValueError:
    return ValueError(
        f"Unsupported model_type {model_type!r}. "
        f"Supported: {', '.join(supported_model_types())}. "
        f"To add support, implement the architecture in tinyinfra/models/ and register it here."
    )
