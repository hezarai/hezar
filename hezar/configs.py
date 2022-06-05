import abc
import os
import copy
from typing import Union, Dict, Literal, Type
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig

from hezar.utils.io import flatten_dict

CONFIG_CLASS = Literal['model', 'trainer', 'dataset']


@dataclass
class BaseConfig:
    config_class: CONFIG_CLASS = field(
        default=None,
        metadata={
            'help': "The category this config is responsible for"
        }
    )

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, os.PathLike], **kwargs):
        """
        Load config from a yaml file
        """
        kwargs = copy.deepcopy(kwargs)
        dict_config = OmegaConf.load(pretrained_path)
        dict_config = cls.from_dict(dict_config, **kwargs)
        return dict_config

    @classmethod
    def from_dict(cls, dict_config: Union[Dict, DictConfig], **kwargs):
        """
        Load config from a dict-like object
        """
        if cls.config_class is not None:
            # load config_class part of the config if config_class is given
            dict_config = dict_config.get(cls.config_class, None)

        dict_config.update(**kwargs)

        for k, v in dict_config.items():
            if not hasattr(cls, k):
                raise ValueError(f'`{cls.__name__}` does not take `{k}` in attributes!\n Hint: add this attribute '
                                 f'to `{cls.__name__}` as:\n `{k}: {v.__class__.__name__} = field(default=None)`')

        config = cls(**dict_config)
        if config is None:
            raise ValueError(f'This dict config has no `{cls.config_class}` key!')
        return config


@dataclass
class ModelConfig(BaseConfig):
    config_class: str = field(default='model')
    name: str = field(
        default=None,
        metadata={
            'help': "Name of the model's class"
        })
    framework: str = field(
        default=None,
        metadata={
            'help': 'ML/DL framework this model is built with, e.g. PyTorch, TensorFlow, etc'
        }
    )
    task: str = field(
        default=None,
        metadata={
            'help': 'The task name this model is built for, e.g. ImageCaptioning, TextClassification, etc.'
        }
    )


@dataclass
class TrainerConfig(BaseConfig):
    config_class = 'trainer'
    name: str = field(
        default=None,
        metadata={
            'help': 'Name of the trainer'
        })
    task: str = field(
        default=None,
        metadata={
            'help': 'The task name this trainer is built for e.g Masked LM, etc.'
        }
    )


@dataclass
class DatasetConfig(BaseConfig):
    config_class = 'dataset'
    name: str = field(
        default=None,
        metadata={
            'help': 'Name of the dataset'
        })
    task: str = field(
        default=None,
        metadata={
            'help': 'Name of the task(s) this dataset is built for'
        }
    )

