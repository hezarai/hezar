import logging
import os
from abc import abstractmethod, ABC
from typing import *

import torch
from torch import nn
from huggingface_hub import HfApi, Repository, hf_hub_download

from hezar.configs import ModelConfig
from hezar.hub import resolve_hub_path, get_local_cache_path
from hezar.hub import HEZAR_TMP_DIR
from hezar.utils import merge_kwargs_into_config, get_logger
from hezar.registry import build_model

logger = get_logger(__name__)


class Model(nn.Module):
    """
    A base model for all models in this library.

    Args:
        config: A dataclass model config
    """
    model_filename = 'model.pt'

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = merge_kwargs_into_config(config, kwargs)

    @classmethod
    def load(cls, hub_or_local_path, load_locally=False, save_to_cache=False, **kwargs):
        """
        Load the model from local path or hub

        Args:
            hub_or_local_path: path to the model living on the Hub or local disk.
            load_locally: force loading from local path
            save_to_cache: Whether to save model and config to Hezar's permanent cache folder

        Returns:
            The fully loaded Hezar model
        """
        # Load config
        config = ModelConfig.load(hub_or_local_path=hub_or_local_path, filename='config.yaml')
        # Build model wih config
        model = build_model(config.name, config, **kwargs)
        # does the path exist locally?
        is_local = load_locally or os.path.isdir(hub_or_local_path)
        if not is_local:
            model_path = hf_hub_download(hub_or_local_path, filename=cls.model_filename, cache_dir=HEZAR_TMP_DIR)
        else:
            model_path = os.path.join(hub_or_local_path, cls.model_filename)
        # Get state dict from the model
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if save_to_cache:
            cache_path = get_local_cache_path(hub_or_local_path, repo_type='model')
            model.save(cache_path)
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
        torch.save(self.state_dict(), os.path.join(path, self.model_filename))
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

    def preprocess(self, inputs, **kwargs):
        """
        Preprocess method intended for use in `self.predict()`. Given raw inputs to the model, perform all the necessary
        preprocessing to get proper inputs for model forward.

        Args:
             inputs: Inputs container e.g, list of texts, list of image paths, etc.
             kwargs: extra arguments specific to the derived class

        Returns:
            Processed input values to feed to the model to forward
        """
        return inputs

    def postprocess(self, inputs, **kwargs):
        """
        Postprocess method intended for use in `self.predict()`. Process model outputs and return human-readable results

        Args:
            inputs: model outputs
            kwargs: extra arguments specific to the derived class

        Returns:
            Processed model output values and converted to human-readable results
        """
        return inputs

    @torch.inference_mode()
    def predict(self, inputs, **kwargs) -> Dict:
        """
        Perform an end-to-end prediction on raw inputs.

        Args:
            inputs: raw inputs e.g, a list of texts, path to images, etc.

        Returns:
            Output dict of results
        """
        self.eval()
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self.forward(model_inputs, **kwargs)
        processed_outputs = self.postprocess(model_outputs, **kwargs)

        return processed_outputs



