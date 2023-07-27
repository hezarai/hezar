import json
import os
from typing import Dict, Union

import omegaconf
from omegaconf import DictConfig

from ..constants import ConfigType, RegistryType
from .common_utils import snake_case
from .logging import get_logger


__all__ = [
    "flatten_dict",
    "load_yaml_config",
    "load_json_config",
    "hezar_config_to_hf_config",
    "get_module_config_class",
    "get_module_class",
    "get_registry_point",
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


def get_registry_point(registry_key: str, registry_type: RegistryType):
    """
    Get the registry item by registry key name in a specific registry

    Args:
        registry_key: Module's name in the registry
        registry_type: Module's registry container type

    Returns:
        A Registry object
    """
    if registry_type == RegistryType.MODEL:
        from ..registry import models_registry  # noqa
        registry = models_registry

    elif registry_type == RegistryType.PREPROCESSOR:
        from ..registry import preprocessors_registry  # noqa
        registry = preprocessors_registry

    elif registry_type == RegistryType.DATASET:
        from ..registry import datasets_registry  # noqa
        registry = datasets_registry

    elif registry_type == RegistryType.EMBEDDING:
        from ..registry import embeddings_registry  # noqa
        registry = embeddings_registry

    elif registry_type == RegistryType.TRAINER:
        from ..registry import trainers_registry  # noqa
        registry = trainers_registry

    elif registry_type == RegistryType.METRIC:
        from ..registry import metrics_registry  # noqa
        registry = metrics_registry

    else:
        raise ValueError(f"Invalid `registry_type`: {registry_type}!")

    registry = registry[registry_key]
    return registry


def get_module_config_class(name: str, config_type: str = None):
    """
    Get the config class for a given module based on its registry name.

    Args:
        name (str): Module's registry name
        config_type (str): Registry type

    Returns:
        A class of type :class:`hezar.Config`
    """
    if config_type == ConfigType.MODEL:
        from ..registry import models_registry  # noqa
        registry = models_registry

    elif config_type == ConfigType.PREPROCESSOR:
        from ..registry import preprocessors_registry  # noqa
        registry = preprocessors_registry

    elif config_type == ConfigType.DATASET:
        from ..registry import datasets_registry  # noqa
        registry = datasets_registry

    elif config_type == ConfigType.EMBEDDING:
        from ..registry import embeddings_registry  # noqa
        registry = embeddings_registry

    elif config_type == ConfigType.METRIC:
        from ..registry import metrics_registry  # noqa
        registry = metrics_registry

    elif config_type == ConfigType.TRAINER:
        from ..registry import trainers_registry  # noqa
        registry = trainers_registry

    elif config_type is None:
        return None
    else:
        raise ValueError(f"Invalid `config_type`: {config_type}!")

    config_cls = registry[name].config_class
    return config_cls


def get_module_class(name: str, module_type: str):
    """
    Get module class based on registry name

    Args:
        name: Module's key name in its registry
        module_type: Type of the module e.g, model, dataset, preprocessor, embedding, etc

    Returns:
        A class corresponding to the given module
    """
    if module_type == ConfigType.MODEL:
        from ..registry import models_registry  # noqa
        registry = models_registry

    elif module_type == ConfigType.PREPROCESSOR:
        from ..registry import preprocessors_registry  # noqa
        registry = preprocessors_registry

    elif module_type == ConfigType.DATASET:
        from ..registry import datasets_registry  # noqa
        registry = datasets_registry

    elif module_type == ConfigType.EMBEDDING:
        from ..registry import embeddings_registry  # noqa
        registry = embeddings_registry

    elif module_type == ConfigType.METRIC:
        from ..registry import metrics_registry  # noqa
        registry = metrics_registry

    elif module_type == ConfigType.TRAINER:
        from ..registry import trainers_registry  # noqa
        registry = trainers_registry

    else:
        raise ValueError(f"Invalid `module_type`: {module_type}!")
    name = snake_case(name)
    module_cls = registry[name].module_class
    return module_cls
