import logging
import os
from abc import abstractmethod, ABC
from typing import *
from enum import Enum

import torch
from torch import nn
from huggingface_hub import HfApi, create_repo
from omegaconf import OmegaConf, DictConfig

from hezar.configs import ModelConfig
from hezar.hub_interface import (HubInterface,
                                 HEZAR_HUB_ID,
                                 HEZAR_MODELS_CACHE_DIR)
from hezar.utils import merge_kwargs_into_config
from hezar.registry import models_registry


class BaseModel(ABC, nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.config = merge_kwargs_into_config(config, kwargs)
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        repo = HubInterface(repo_name_or_id=path, repo_type='model')
        model_name = repo.get_model_registry_name()
        model_config_class = models_registry[model_name]['model_config']
        config = repo.get_config(model_config_class=model_config_class)
        model = cls(config, mode='inference', repo=repo, **kwargs)
        model.load_state_dict(repo.get_model(return_state_dict=True))
        return model

    def save_pretrained(self, path):
        # TODO handle the case when files exist
        # save config
        self.config.save_pretrained(path)
        # save model
        model_state_dict = self.model.state_dict()
        save_path = os.path.join(path, 'pytorch.bin')
        torch.save(model_state_dict, save_path)
        logging.info(f'Saved model weights to `{save_path}`')

    def push_to_hub(self, repo_name_or_id):
        repo = HubInterface(repo_name_or_id=repo_name_or_id, repo_type='model')
        self.save_pretrained(repo.repo_dir)
        repo.push_to_hub('Upload from Hezar!')

    @abstractmethod
    def forward(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs, **kwargs) -> Dict:
        raise NotImplementedError



