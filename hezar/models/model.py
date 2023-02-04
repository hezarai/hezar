import logging
import os
from abc import abstractmethod, ABC
from typing import *

import torch
from torch import nn

from hezar.configs import ModelConfig
from hezar.hub_interface import (HubInterface,
                                 HEZAR_HUB_ID,
                                 HEZAR_MODELS_CACHE_DIR)
from hezar.utils import merge_kwargs_into_config, get_logger
from hezar.registry import models_registry

logger = get_logger('hezar.model')


class Model(nn.Module):
    """
    A base model for all models in this library.

    Args:
        config: A dataclass model config
    """
    def __init__(self,
                 config,
                 **kwargs):
        super(Model, self).__init__()
        self.config = merge_kwargs_into_config(config, kwargs)
        self.model: nn.Module = self.build_model()

    @abstractmethod
    def build_model(self):
        """
        Build the model using the properties in `self.config`. This method is only responsible for building the model
        architecture. No weights loading is necessary here.

        Returns:
            A :class:`nn.Module` instance
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load the model from local path or hub

        Args:
            path: path to the model living on the Hub or local disk.

        Returns:
            The fully loaded Hezar model
        """
        repo = HubInterface(repo_name_or_id=path, repo_type='model')
        model_name = repo.get_model_registry_name()
        model_config_class = models_registry[model_name]['model_config']
        config = repo.get_config(model_config_class=model_config_class)
        model = cls(config, mode='inference', repo=repo, **kwargs)
        model.load_state_dict(repo.get_model(return_state_dict=True))
        return model

    def load_state_dict(self, state_dict, **kwargs):
        try:
            super().load_state_dict(state_dict, strict=True)
        except RuntimeError:
            super().load_state_dict(state_dict, strict=False)
            logger.warning(f"Partially loading the weights as the model architecture and the given state dict are "
                           f"incompatible! \nIgnore this warning in case you plan on fine-tuning this model")

    def save(self, path: Union[str, os.PathLike]):
        """
        Save model weights and config to a local path

        Args:
            path: A local directory to save model, config, etc.
        """
        # TODO handle the case when files exist
        # save config
        self.config.save(path)
        # save model
        model_state_dict = self.model.state_dict()
        save_path = os.path.join(path, 'pytorch.bin')
        torch.save(model_state_dict, save_path)
        logging.info(f'Saved model weights to `{save_path}`')

    def push_to_hub(self, repo_name_or_id):
        """
        Push the model and required files to the hub

        Args:
            repo_name_or_id: The path (id or repo name) on the hub
        """
        repo = HubInterface(repo_name_or_id=repo_name_or_id, repo_type='model')
        self.save(repo.repo_dir)
        repo.push_to_hub('Upload from Hezar!')

    @abstractmethod
    def forward(self, inputs, **kwargs) -> Dict:
        """
        Forward inputs through the model and return logits, etc.

        Args:
            inputs: The required inputs for the model forward

        Returns:
            A dict of outputs like logits, loss, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs, **kwargs) -> Dict:
        """
        Perform an end-to-end prediction on raw inputs.

        Args:
            inputs: raw inputs e.g, a list of texts, path to images, etc.

        Returns:
            Output dict of results
        """
        raise NotImplementedError



