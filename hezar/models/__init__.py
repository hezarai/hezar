import importlib
import os
from typing import *

from .model import Model, build_model
from hezar.registry import models_registry


def register_model(model_name: str, model_config):
    def register_model_class(cls):
        if model_name in models_registry:
            raise ValueError(f'Requested model `{model_name}` already exists_on_hub in the registry!')

        model_config.name = model_name
        models_registry[model_name] = dict(model_class=cls, model_config=model_config)

        return cls

    return register_model_class


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
