from dataclasses import dataclass, field

from omegaconf import DictConfig

from hezar.configs import ModelConfig


@dataclass
class DistilBertTextClassificationConfig(ModelConfig):
    name = 'TextClassificationDistilBert'
    pretrained_path: str = ...
    hf_model_config: DictConfig = None
    framework = 'pt'
    task = 'text classification'
    vocab_size: int = field(default=10000)
