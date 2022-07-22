import abc
from typing import Dict, Type

from ..configs import ModelConfig
from ..utils import merge_kwargs_into_config


class BaseModel(abc.ABC):
    def __init__(self, config: ModelConfig, **kwargs):
        super(BaseModel, self).__init__()
        self.config = merge_kwargs_into_config(config, kwargs)
        self.model = self.build_model()

    def __str__(self):
        return self.model

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, path, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError
