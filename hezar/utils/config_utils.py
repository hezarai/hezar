import json
import os
from typing import Dict, Union

import omegaconf
from omegaconf import DictConfig

from .logging import get_logger

__all__ = [
    "merge_kwargs_into_config",
    "flatten_dict",
    "load_yaml_config",
    "load_json_config",
    "hezar_config_to_hf_config",
    "get_model_config_class",
]

logger = get_logger(__name__)


def merge_kwargs_into_config(config, **kwargs):
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            logger.warning(f"{str(config.__class__.__name__)} does not take `{k}` as a config parameter!")

    return config


def flatten_dict(dict_config: Union[Dict, DictConfig]) -> DictConfig:
    """
    Flatten a nested Dict/DictConfig object

    Args:
        dict_config: A Dict/DictConfig object

    Returns:
        The flattened version of the dict-like object
    """

    config = DictConfig(dict())
    for k, v in dict_config.items():
        if isinstance(v, (Dict, DictConfig)):
            config.update(flatten_dict(v))
        else:
            config[k] = v

    return config


def load_yaml_config(path: Union[str, os.PathLike]):
    """
    Load yaml config file using omegaconf
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


def get_model_config_class(name: str):
    """
    Get the config class for a given model based on its registry name.

    Args:
        name (str): model's registry name

    Returns:
        A class of type :class:`hezar.Config`
    """
    from ..registry import models_registry
    config_cls = models_registry[name]["config_class"]
    return config_cls
