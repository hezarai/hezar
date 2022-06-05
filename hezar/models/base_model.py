import abc
from typing import Dict, Type

from hezar.configs import ModelConfig


class BaseModel(metaclass=abc.ABC):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.model = self.build_model()
        self.__dict__.update(kwargs)

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
