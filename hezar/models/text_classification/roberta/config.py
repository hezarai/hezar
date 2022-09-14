from dataclasses import dataclass, field

from omegaconf import DictConfig

from hezar.configs import ModelConfig


@dataclass
class RobertaTextClassificationConfig(ModelConfig):
    name: str = 'RobertaTextClassification'
    pretrained_path: str = ...
    hft_model_config: DictConfig = None
    framework = 'pt'
    task = 'text classification'
    vocab_size: int = field(default=10000)
