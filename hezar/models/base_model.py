from abc import abstractmethod, ABC
from typing import Dict

from ..configs import ModelConfig
from ..utils import merge_kwargs_into_config


class BaseModel(ABC):
    def __init__(self, config: ModelConfig, **kwargs):
        super(BaseModel, self).__init__()
        self.config = merge_kwargs_into_config(config, kwargs)
        self.model = self.build_model()

    def __str__(self):
        return self.model.__str__()

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def push_to_hub(self, path, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def train_batch(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate_batch(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError
