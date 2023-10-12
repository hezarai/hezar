from dataclasses import dataclass, field

from ....configs import ModelConfig
from ....constants import TaskType


@dataclass
class DistilBertSequenceLabelingConfig(ModelConfig):
    name = "distilbert_sequence_labeling"
    task: str = TaskType.SEQUENCE_LABELING
    num_labels: int = None
    id2label: dict = None
    activation: str = "gelu"
    attention_dropout: float = 0.1
    dim: int = 768
    dropout: float = 0.1
    initializer_range: float = 0.02
    max_position_embeddings: int = 512
    n_heads: int = 12
    n_layers: int = 6
    output_past: bool = True
    pad_token_id: int = 0
    qa_dropout: float = 0.1
    tie_weights_: bool = True
    vocab_size: int = 42000

    hidden_dropout_prob: float = 0.1
    use_cache: bool = True
    classifier_dropout: float = None
    prediction_skip_tokens: list = field(default_factory=lambda: ["[SEP]", "[CLS]"])
