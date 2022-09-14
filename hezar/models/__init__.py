import importlib
import os

from omegaconf import OmegaConf, DictConfig

from .base_model import BaseModel
from ..configs import ModelConfig

__all__ = [
    "BaseModel",
    "ModelConfig",
    "models_registry",
    "register_model"
]

models_registry = {}


def register_model(model_name, model_config):
    def register_model_class(cls):
        if model_name in models_registry:
            raise ValueError(f'Requested model `{model_name}` already exists in the registry!')
        if not issubclass(cls, BaseModel):
            raise ValueError(f'The model class for `{model_name}: {cls.__name__}` must extend `BaseModel`!')
        if not issubclass(model_config, ModelConfig):
            raise ValueError(
                f'The model config for `{model_config}: {model_config.__name__}` must extend `ModelConfig`!')

        models_registry[model_name] = dict(model_class=cls, model_config=model_config)

        return cls

    return register_model_class


def import_models(models_dir, namespace):
    tasks_dir = os.listdir(models_dir)
    for module in tasks_dir:
        path = os.path.join(models_dir, module)
        if not module.startswith('_') and not module.startswith('.') and os.path.isdir(path):
            for file in os.listdir(path):
                if (
                        not file.startswith("_")
                        and not file.startswith(".")
                        and (file.endswith(".py") or os.path.isdir(path))
                ):
                    model_name = module[: module.find(".py")] if file.endswith(".py") else file
                    importlib.import_module(f'{namespace}.{module}.{model_name}')


import_models(os.path.dirname(__file__), "hezar.models")
