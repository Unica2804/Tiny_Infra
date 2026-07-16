from dataclasses import dataclass

@dataclass
class QwenConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    num_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6