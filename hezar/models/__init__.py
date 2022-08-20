import importlib
import os

from .base_model import BaseModel
from ..configs import ModelConfig

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
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)


import_models(os.path.dirname(__file__), "hezar.models")
