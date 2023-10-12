from dataclasses import dataclass

from ....configs import ModelConfig


@dataclass
class T5TextGenerationConfig(ModelConfig):
    name = "t5_text_generation"
    vocab_size: int = 32103
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 12
    num_decoder_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    decoder_start_token_id: int = 0
    eos_token_id: int = 1
    min_length: int = 0
    max_length: int = 100
