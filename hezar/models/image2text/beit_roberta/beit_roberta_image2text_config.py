from dataclasses import dataclass, field

from ....configs import ModelConfig


@dataclass
class DecoderConfig(ModelConfig):
    name = "beit_roberta_decoder"
    is_decoder: bool = True
    add_cross_attention: bool = True
    attention_probs_dropout_prob: float = 0.1
    bos_token_id: int = 0
    eos_token_id: int = 2
    classifier_dropout: float = None
    gradient_checkpointing: bool = False
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 768
    initializer_range: int = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 514
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    pad_token_id: int = 1
    position_embedding_type: str = "absolute"
    type_vocab_size: int = 1
    use_cache: bool = True
    vocab_size: int = 42000


@dataclass
class EncoderConfig(ModelConfig):
    name = "beit_roberta_encoder"
    vocab_size = 8192
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    hidden_act = "gelu"
    hidden_dropout_prob = 0.0
    attention_probs_dropout_prob = 0.0
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    image_size = 224
    patch_size = 16
    num_channels = 3
    use_mask_token = False
    use_absolute_position_embeddings = False
    use_relative_position_bias = False
    use_shared_relative_position_bias = False
    layer_scale_init_value = 0.1
    drop_path_rate = 0.1
    use_mean_pooling = True
    out_indices = [3, 5, 7, 11]
    pool_scales = [1, 2, 3, 6]
    use_auxiliary_head = True
    auxiliary_loss_weight = 0.4
    auxiliary_channels = 256
    auxiliary_num_convs = 1
    auxiliary_concat_input = False
    semantic_loss_ignore_index = 255


@dataclass
class GenerationConfig:
    bos_token_id: int = 0
    decoder_start_token_id: int = 0
    early_stopping: bool = True
    eos_token_id: int = 2
    length_penalty: float = 2.0
    max_length: int = 64
    no_repeat_ngram_size: int = 3
    num_beams: int = 4
    pad_token_id: int = 1


@dataclass
class BeitRobertaImage2TextConfig(ModelConfig):
    name = "beit_roberta_image2text"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
