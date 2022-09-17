from dataclasses import dataclass, field

from omegaconf import DictConfig

from hezar.configs import ModelConfig


@dataclass
class DistilBertTextClassificationConfig(ModelConfig):
    name: str = None  # initialized on registry
    inner_model_config: DictConfig = None
