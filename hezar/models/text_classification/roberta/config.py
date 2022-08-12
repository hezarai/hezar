from dataclasses import dataclass, field

from hezar.configs import ModelConfig


@dataclass
class RobertaTextClassificationConfig(ModelConfig):
    name = 'TextClassificationRoberta'
    pretrained_path: str = 'hezar-ai/roberta-fa-zwnj-base-classification-sentiment'
    framework = 'pt'
    task = 'text classification'
    vocab_size: int = field(default=10000)
