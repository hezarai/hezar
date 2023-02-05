from abc import ABC, abstractmethod

from hezar.configs import TrainConfig
from hezar.models import models_registry
from hezar.registry import criterions_registry, optimizers_registry


class Trainer(ABC):
    def __init__(self, config: TrainConfig, **kwargs):
        self.config = config
        self.device = self.config.device
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config
        self.criterion_config = self.config.criterion_config
        self.optimizer_config = self.config.optimizer_config

        self.model = self.build_model()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.data_loaders = self.build_dataloaders()

    def build_dataloaders(self):
        raise NotImplementedError

    def build_model(self):
        model_name = self.model_config.name
        model = models_registry[model_name](self.model_config)
        return model

    def build_criterion(self):
        criterion = criterions_registry[self.criterion_config.name](**self.criterion_config.dict())
        criterion.to(self.device)
        return criterion

    def build_optimizer(self):
        optimizer = optimizers_registry[self.optimizer_config.name](**self.optimizer_config.dict())
        return optimizer

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError
