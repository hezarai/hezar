import importlib
import os

from .preprocessor import Preprocessor
from hezar.registry import preprocessors_registry

__all__ = [
    "Preprocessor",
    "preprocessors_registry",
    "register_preprocessor",
    "import_preprocessors",
]


def register_preprocessor(preprocessor_name: str, config_class):
    def register_preprocessor_class(cls):
        if preprocessor_name in preprocessors_registry:
            raise ValueError(f"Requested preprocessor `{preprocessor_name}` already exists in the registry!")

        config_class.name = preprocessor_name
        preprocessors_registry[preprocessor_name] = dict(preprocessor_class=cls, config_class=config_class)

        return cls

    return register_preprocessor_class


def import_preprocessors(preprocessors_dir, namespace):
    for module in os.listdir(preprocessors_dir):
        path = os.path.join(preprocessors_dir, module)
        if (
            not module.startswith("_")
            and not module.startswith(".")
            and (module.endswith(".py") or os.path.isdir(path))
        ):
            preprocessor_name = module[: module.find(".py")] if module.endswith(".py") else module
            importlib.import_module(namespace + "." + preprocessor_name)


# import all preprocessors in the `preprocessors` module so that their classes are registered
import_preprocessors(os.path.dirname(__file__), "hezar.preprocessors")
