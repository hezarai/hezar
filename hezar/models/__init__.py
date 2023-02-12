import importlib
import os

from .model import Model
from hezar.registry import models_registry, build_model

__all__ = ["Model", "register_model", "models_registry", "import_models", "build_model"]


def register_model(model_name: str, config_class):
    def register_model_class(cls):
        if model_name in models_registry:
            raise ValueError(f"Requested model `{model_name}` already exists in the registry!")

        config_class.name = model_name
        models_registry[model_name] = dict(model_class=cls, config_class=config_class)

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


# import all models in the `models` module so that their classes are registered
import_models(os.path.dirname(__file__), "hezar.models")
