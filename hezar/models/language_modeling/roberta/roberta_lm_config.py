from dataclasses import dataclass

from ....configs import ModelConfig
from ....constants import TaskType


@dataclass
class RobertaLMConfig(ModelConfig):
    name = "roberta_lm"
    task: str = TaskType.LANGUAGE_MODELING
    attention_probs_dropout_prob: float = 0.1
    bos_token_id: int = 0
    eos_token_id: int = 2
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
