import logging
import os
from abc import abstractmethod, ABC
from typing import *

import torch
from torch import nn
from huggingface_hub import HfApi, Repository

from hezar.configs import ModelConfig
from hezar.hub import clone_repo, resolve_hub_path, get_local_cache_path, HEZAR_MODELS_CACHE_DIR
from hezar.utils import merge_kwargs_into_config, get_logger
from hezar.registry import models_registry

logger = get_logger('hezar.model')


class Model(nn.Module):
    """
    A base model for all models in this library.

    Args:
        config: A dataclass model config
    """

    def __init__(self, config, **kwargs):
        super().__init__()
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
        # clone full repo from hub
        repo = clone_repo(hub_path=path, save_path=HEZAR_MODELS_CACHE_DIR)
        # Load config
        config = ModelConfig.load(path=repo, filename='config.yaml')
        # Build model wih config
        model = load_model(config.name, config, **kwargs)
        # Get state dict from the model in the cloned repo
        state_dict = torch.load(f'{repo}/pytorch.bin')
        model.load_state_dict(state_dict)
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
        # save model and config to the repo
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f'{path}/pytorch.bin')
        self.config.save(save_dir=path, filename='config.yaml')
        logger.info(f'Saved model and config to `{path}`')

    def push_to_hub(self, hub_path):
        """
        Push the model and required files to the hub

        Args:
            hub_path: The path (id or repo name) on the hub
        """
        api = HfApi()
        repo_id = resolve_hub_path(hub_path)
        # create remote repo
        api.create_repo(repo_id, repo_type='model', exist_ok=True)
        # create local repo
        cache_path = get_local_cache_path(hub_path, repo_type='model')
        repo = Repository(local_dir=cache_path, clone_from=repo_id)
        self.save(cache_path)
        repo.push_to_hub(f'Hezar: Upload {self.config.name}')
        logger.info(f'Model successfully pushed to `{repo_id}`')

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


def load_model(name, config=None, **kwargs):
    """
    Given the name of the model (in the registry), load the model. If config is None then the model will be loaded using
    the default config.

    Args:
        name: name of the model in the models' registry
        config: a ModelConfig instance
        kwargs: extra config parameters that is loaded to the model
    """
    config = config or models_registry[name]['model_config']()
    model = models_registry[name]['model_class'](config, **kwargs)
    return model
