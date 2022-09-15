import os
import copy
import os
from dataclasses import dataclass, field, asdict
from typing import *

import torch
from torch import Tensor
from omegaconf import DictConfig

from hezar.utils.hub_utils import load_config_from_hub

CONFIG_CLASS = Literal['base', 'model', 'dataset', 'task']


@dataclass
class BaseConfig:
    config_class: CONFIG_CLASS = field(
        default='base',
        metadata={
            'help': "The category this config is responsible for"
        }
    )
    dict = asdict

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, os.PathLike], filename='config.yaml', **kwargs):
        """
        Load config from Hub or locally if it already exists (handled by HfApi)
        """
        kwargs = copy.deepcopy(kwargs)
        config = load_config_from_hub(pretrained_path, filename=filename)
        if cls.config_class != 'base':
            # if config_class is not `base` create a {config_class}Config() instance e.g, ModelConfig()
            config = cls.from_dict(config, **kwargs)
        return config

    @classmethod
    def from_dict(cls, dict_config: Union[Dict, DictConfig], strict=False, **kwargs):
        """
        Load config from a dict-like object
        """
        # load config_class part of the config if config_class is given
        dict_config = dict_config[cls.config_class]
        dict_config.update(**kwargs)

        for k, v in dict_config.items():
            if not hasattr(cls, k):
                if strict:
                    raise ValueError(f'`{cls.__name__}` does not take `{k}` in attributes!\n Hint: add this attribute '
                                     f'to `{cls.__name__}` as:\n `{k}: {v.__class__.__name__} = field(default=None)`')
                else:
                    setattr(cls, k, v)

        config = cls(**dict_config)
        if config is None:
            raise ValueError(f'This dict config has no `{cls.config_class}` key!')
        return config


@dataclass
class ModelConfig(BaseConfig):
    config_class = 'model'
    name: str = field(
        default=None,
        metadata={
            'help': "Name of the model's key in the models_registry"
        })


@dataclass
class DatasetConfig(BaseConfig):
    config_class = 'dataset'
    name: str = field(
        default=None,
        metadata={
            'help': 'Name of the dataset'
        })
    task: Union[str, List[str]] = field(
        default=None,
        metadata={
            'help': 'Name of the task(s) this dataset is built for'
        }
    )


@dataclass
class CriterionConfig(BaseConfig):
    name: str = None
    weight: Optional[Tensor] = None
    reduce: str = None
    ignore_index: int = -100


@dataclass
class OptimizerConfig(BaseConfig):
    name: str = None
    lr: float = None


@dataclass
class TaskConfig(BaseConfig):
    config_class = 'task'
    device: str = 'cpu'
    model_name: str = field(
        default=None,
        metadata={
            'help': 'name of the model in the models_registry'
        })
    name: str = field(
        default=None,
        metadata={
            'help': 'Name of the task'
        })
    model_config: ModelConfig = field(
        default=ModelConfig(),
        metadata={
            'help': 'model config for this task'
        })
    dataset_config: DatasetConfig = field(
        default=DatasetConfig(),
        metadata={
            'help': 'dataset config for this task'
        }
    )
    criterion_config: CriterionConfig = field(
        default=CriterionConfig(),
        metadata={
            'help': 'criterion config for this task'
        })
    optimizer_config: OptimizerConfig = field(
        default=OptimizerConfig(),
        metadata={
            'help': 'optimizer config for this task'
        })
