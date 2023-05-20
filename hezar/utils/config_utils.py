import json
import os
from typing import Dict, Union

import omegaconf
from omegaconf import DictConfig

from .logging import get_logger


__all__ = [
    "flatten_dict",
    "load_yaml_config",
    "load_json_config",
    "hezar_config_to_hf_config",
    "get_module_config_class",
]

logger = get_logger(__name__)


def flatten_dict(dict_config: Union[Dict, DictConfig]) -> DictConfig:
    """
    Flatten a nested Dict/DictConfig object

    Args:
        dict_config: A Dict/DictConfig object

    Returns:
        The flattened version of the dict-like object
    """

    config = DictConfig({})
    for k, v in dict_config.items():
        if isinstance(v, (Dict, DictConfig)):
            config.update(flatten_dict(v))
        else:
            config[k] = v

    return config


def load_yaml_config(path: Union[str, os.PathLike]):
    """
    Load yaml file using omegaconf
    """
    config = omegaconf.OmegaConf.load(path)
    return config


def load_json_config(path: Union[str, os.PathLike]):
    """
    Load json config file
    """
    with open(path) as f:
        config = json.load(f)
    return config


def hezar_config_to_hf_config(config):
    """
    Convert a :class:`hezar.Config` instance to HuggingFace :class:`transformers.PretrainedConfig` format

    Args:
        config: A :class:`Hezar.Config` instance
    """
    from transformers import PretrainedConfig

    hf_config = PretrainedConfig(**config)
    return hf_config


def get_module_config_class(name: str, config_type: str):
    """
    Get the config class for a given module based on its registry name.

    Args:
        name (str): Model's registry name
        config_type (str): Registry type

    Returns:
        A class of type :class:`hezar.Config`
    """
    if config_type == "model":
        from ..registry import models_registry  # noqa

        registry = models_registry
    elif config_type == "preprocessor":
        from ..registry import preprocessors_registry  # noqa

        registry = preprocessors_registry
    elif config_type == "dataset":
        from ..registry import datasets_registry  # noqa

        registry = datasets_registry
    elif config_type == "embedding":
        from ..registry import embeddings_registry
        registry = embeddings_registry

    else:
        raise ValueError(f"Invalid `config_type`: {config_type}!")

    config_cls = registry[name]["config_class"]
    return config_cls
