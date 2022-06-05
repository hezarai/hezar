from dataclasses import dataclass, field

from hezar.configs import ModelConfig


@dataclass
class DistilBertTextClassificationConfig(ModelConfig):
    name = 'TextClassificationDistilBert'
    pretrained_path: str = ''
    framework = 'pt'
    task = 'text classification'
    vocab_size: int = field(default=10000)

