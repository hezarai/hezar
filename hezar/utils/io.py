import json
import os
from typing import Dict, Union
import json

import omegaconf
from omegaconf import DictConfig


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
