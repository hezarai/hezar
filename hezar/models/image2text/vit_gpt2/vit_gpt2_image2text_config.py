from dataclasses import dataclass, field

from ....configs import ModelConfig


@dataclass
class DecoderConfig(ModelConfig):
    name = "vit_gpt2_decoder"
    add_cross_attention: bool = True
    vocab_size: int = 42001
    attn_pdrop: float = 0.1
    bos_token_id: int = 5
    embd_pdrop: float = 0.1
    eos_token_id: int = 5
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05
    model_type: str = "gpt2"
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_inner: int = None
    n_layer: int = 12
    n_positions: int = 1024
    resid_pdrop: float = 0.1
    summary_activation: bool = False
    summary_first_dropout: float = 0.1
    use_cache: bool = True


@dataclass
class EncoderConfig(ModelConfig):
    name = "vit_gpt2_encoder"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    encoder_stride: int = 16


@dataclass
class GenerationConfig(ModelConfig):
    bos_token_id: int = 0
    decoder_start_token_id: int = 0
    early_stopping: bool = True
    eos_token_id: int = 2
    length_penalty: float = 2.0
    max_new_tokens: int = 24
    no_repeat_ngram_size: int = 3
    num_beams: int = 4
    pad_token_id: int = 1


@dataclass
class ViTGPT2Image2TextConfig(ModelConfig):
    name = "vit_gpt2_image2text"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
