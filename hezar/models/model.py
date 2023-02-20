import os
from abc import abstractmethod
from typing import *

import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

from hezar.configs import ModelConfig
from hezar.constants import HEZAR_TMP_DIR, DEFAULT_MODEL_FILENAME, DEFAULT_CONFIG_FILENAME
from hezar.hub_utils import resolve_hub_path, get_local_cache_path
from hezar.registry import build_model
from hezar.utils import get_logger

logger = get_logger(__name__)


class Model(nn.Module):
    """
    A base model for all models in this library.

    Args:
        config: A dataclass model config
    """

    model_filename = DEFAULT_MODEL_FILENAME
    config_filename = DEFAULT_CONFIG_FILENAME

    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__()
        self.config = config.update(**kwargs)

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
        hub_or_local_path = resolve_hub_path(hub_or_local_path)
        # Load config
        config = ModelConfig.load(hub_or_local_path=hub_or_local_path, filename="config.yaml")
        # Build model wih config
        model = build_model(config.name, config, **kwargs)
        # does the path exist locally?
        is_local = load_locally or os.path.isdir(hub_or_local_path)
        if not is_local:
            model_path = hf_hub_download(
                hub_or_local_path,
                filename=model.model_filename,
                cache_dir=HEZAR_TMP_DIR,
                resume_download=True,
            )
        else:
            model_path = os.path.join(hub_or_local_path, model.model_filename)
        # Get state dict from the model
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if save_to_cache:
            cache_path = get_local_cache_path(hub_or_local_path, repo_type="model")
            model.save(cache_path)
        return model

    def load_state_dict(self, state_dict, **kwargs):
        try:
            super().load_state_dict(state_dict, strict=True)
        except RuntimeError:
            super().load_state_dict(state_dict, strict=False)
            logger.warning(
                f"Partially loading the weights as the model architecture and the given state dict are "
                f"incompatible! \nIgnore this warning in case you plan on fine-tuning this model"
            )

    def save(self, path: Union[str, os.PathLike], save_config: bool = True):
        """
        Save model weights and config to a local path

        Args:
            path: A local directory to save model, config, etc.
            save_config (bool): Whether to save config along with the model or not.

        Returns:
            Path to the saved model
        """
        # save model and config to the repo
        os.makedirs(path, exist_ok=True)
        model_save_path = os.path.join(path, self.model_filename)
        torch.save(self.state_dict(), model_save_path)
        if save_config:
            self.config.save(save_dir=path, filename=self.config_filename)
        logger.info(f"Saved model weights to `{path}`")
        return model_save_path

    def push_to_hub(self, hub_path, commit_message=None, private=False):
        """
        Push the model and required files to the hub

        Args:
            hub_path: The path (id or repo name) on the hub
            commit_message (str): Commit message for this push
            private (bool): Whether to create a private repo or not
        """
        api = HfApi()
        repo_id = resolve_hub_path(hub_path)
        # create remote repo
        repo_url = api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        logger.info(f"Prepared repo `{repo_url}`. Starting push process...")
        # create local repo
        cache_path = get_local_cache_path(hub_path, repo_type="model")
        model_save_path = self.save(cache_path, save_config=False)
        if commit_message is None:
            commit_message = f"Hezar: Upload model and config"
        # upload config file
        self.config.push_to_hub(
            repo_id, filename=self.config_filename, repo_type="model", commit_message=commit_message
        )
        # upload model file
        logger.info(f"Pushing model file: `{self.model_filename}`")
        api.upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo=self.model_filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Uploaded `{model_save_path}` to `{repo_id}` as `{self.model_filename}`")

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

    def post_process(self, inputs, **kwargs):
        """
        Process model outputs and return human-readable results. Called in `self.predict()`

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
        model_outputs = self.forward(inputs, **kwargs)
        processed_outputs = self.post_process(model_outputs, **kwargs)

        return processed_outputs
