from dataclasses import asdict, dataclass, field
from typing import List

from ....configs import ModelConfig


@dataclass
class WhisperSpeechRecognitionGenerationConfig:
    alignment_heads: List[List[int]] = None
    begin_suppress_tokens: List[int] = field(default_factory=lambda: [220, 50256])
    bos_token_id: int = 50257
    decoder_start_token_id: int = 50258
    eos_token_id: int = 50257
    forced_decoder_ids: List[List[int]] = field(default_factory=lambda: [[1, None], [2, 50359]])
    is_multilingual: bool = True
    max_initial_timestamp_index: int = 50
    max_length: int = 448
    no_timestamps_token_id: int = 50363
    pad_token_id: int = 50257
    prev_sot_token_id: int = 50361
    return_timestamps: int = False
    suppress_tokens: List[int] = None
    task_to_id: dict[str, int] = field(default_factory=lambda: {"transcribe": 50359, "translate": 50358})

    def dict(self):
        return asdict(self)


@dataclass
class WhisperSpeechRecognitionConfig(ModelConfig):
    name = "whisper_speech_recognition"
    vocab_size: int = 51865
    num_mel_bins: int = 80
    encoder_layers: int = 6
    encoder_attention_heads: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 4
    num_hidden_layers: int = 12
    decoder_ffn_dim: int = 1536
    encoder_ffn_dim: int = 1536
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    decoder_start_token_id: int = 50257
    use_cache: bool = True
    sampling_rate: int = 16000
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 256
    dropout: float = 0.0
    torch_dtype: str = "float32"
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    init_std: float = 0.02
    scale_embedding: bool = False
    max_source_positions: int = 1500
    max_target_positions: int = 448
    pad_token_id: int = 50256
    bos_token_id: int = 50257
    eos_token_id: int = 50256
    suppress_tokens: List[int] = None
    begin_suppress_tokens: List[int] = field(default_factory=lambda: [220, 50256])
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    apply_spec_augment: bool = False
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    max_new_tokens: int = 448
    generation_config: dict | WhisperSpeechRecognitionGenerationConfig = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.generation_config, dict):
            self.generation_config = WhisperSpeechRecognitionGenerationConfig(**self.generation_config)
        elif self.generation_config is None:
            self.generation_config = WhisperSpeechRecognitionGenerationConfig()
