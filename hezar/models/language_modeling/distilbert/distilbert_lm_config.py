from dataclasses import dataclass

from ....configs import ModelConfig
from ....constants import TaskType


@dataclass
class DistilBertLMConfig(ModelConfig):
    name = "distilbert_lm"
    task: str = TaskType.LANGUAGE_MODELING
    activation: str = "gelu"
    attention_dropout: float = 0.1
    dim: int = 768
    dropout: float = 0.1
    hidden_dim: int = 3072
    initializer_range: float = 0.02
    max_position_embeddings: int = 512
    n_heads: int = 12
    n_layers: int = 6
    output_past: bool = True
    pad_token_id: int = 0
    qa_dropout: float = 0.1
    tie_weights_: bool = True
    vocab_size: int = 42000
