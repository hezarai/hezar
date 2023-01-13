import importlib
import os
from typing import *

from .base_model import BaseModel
from hezar.registry import models_registry


def register_model(model_name: str, model_config):
    def register_model_class(cls):
        if model_name in models_registry:
            raise ValueError(f'Requested model `{model_name}` already exists in the registry!')

        model_config.name = model_name
        models_registry[model_name] = dict(model_class=cls, model_config=model_config)

        return cls

    return register_model_class


def load_model(name, mode='training', **kwargs):
    config = models_registry[name]['model_config'](**kwargs)
    model = models_registry[name]['model_class'](config, mode=mode)
    return model


def import_models(models_dir, namespace):
    for module in os.listdir(models_dir):
        path = os.path.join(models_dir, module)
        if (
                not module.startswith("_")
                and not module.startswith(".")
                and (module.endswith(".py") or os.path.isdir(path))
        ):
            model_name = module[: module.find(".py")] if module.endswith(".py") else module
            importlib.import_module(namespace + "." + model_name)


import_models(os.path.dirname(__file__), "hezar.models")
