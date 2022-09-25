from dataclasses import dataclass, field
from typing import *

from omegaconf import DictConfig

from hezar.configs import ModelConfig


@dataclass
class DistilBertTextClassificationConfig(ModelConfig):
    name: str = None  # initialized on registry
    inner_model_config: DictConfig = None
    invalid_chars: List = field(default_factory=list)
