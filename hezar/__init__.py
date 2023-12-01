"""
Direct importing from hezar's root is no longer supported nor recommended since version 0.33.0. The following is just a
workaround for backward compatibility. Any class, functions, etc. must be imported from its main submodule under hezar.
"""
import warnings


__version__ = "0.34.0"


def _warn_on_import(name: str, submodule: str):
    warnings.warn(
        f"Importing {name} from hezar root is deprecated and will be removed soon. "
        f"Please use `from {submodule} import {name}`"
    )


def __getattr__(name: str):
    if name == "Model":
        from hezar.models import Model
        _warn_on_import(name, "hezar.models")
        return Model
    elif name == "Dataset":
        from .data import Dataset
        _warn_on_import(name, "hezar.data")
        return Dataset
    elif name == "Trainer":
        from .trainer import Trainer
        _warn_on_import(name, "hezar.trainer")
        return Trainer
    elif name == "Embedding":
        from .embeddings import Embedding
        _warn_on_import(name, "hezar.embeddings")
        return Embedding
    elif name == "Preprocessor":
        from .preprocessors import Preprocessor
        _warn_on_import(name, "hezar.preprocessors")
        return Preprocessor
    elif name == "Metric":
        from .metrics import Metric
        _warn_on_import(name, "hezar.metrics")
        return Metric
    elif "Config" in name:
        from .configs import Config
        _warn_on_import(name, "hezar.configs")
        return Config


__all__ = [
    "Config",
    "Model",
    "Dataset",
    "Trainer",
    "Preprocessor",
    "Embedding",
    "Metric",
]
