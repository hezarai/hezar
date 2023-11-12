from dataclasses import dataclass, field

from ....configs import ModelConfig


@dataclass
class GenerationConfig(ModelConfig):
    bos_token_id: int = 0
    decoder_start_token_id: int = 0
    early_stopping: bool = True
    eos_token_id: int = 2
    length_penalty: float = 2.0
    max_new_tokens: int = 50
    no_repeat_ngram_size: int = 3
    num_beams: int = 4
    pad_token_id: int = 1


@dataclass
class GPT2TextGenerationConfig(ModelConfig):
    name = "gpt2_text_generation"
    add_cross_attention: bool = False
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
    generation: GenerationConfig = field(default_factory=GenerationConfig)
