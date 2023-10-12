from dataclasses import dataclass, field

from ....configs import ModelConfig
from ....constants import TaskType


@dataclass
class BertSequenceLabelingConfig(ModelConfig):
    name = "bert_sequence_labeling"
    task: str = TaskType.SEQUENCE_LABELING
    num_labels: int = None
    id2label: dict = None
    vocab_size: int = 42000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: float = None
    prediction_skip_tokens: list = field(default_factory=lambda: ["[SEP]", "[CLS]"])
