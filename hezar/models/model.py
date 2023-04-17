import os
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Union

import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

from ..builders import build_model
from ..configs import ModelConfig
from ..constants import DEFAULT_MODEL_CONFIG_FILE, DEFAULT_MODEL_FILE, HEZAR_CACHE_DIR
from ..utils import get_logger
from ..utils.hub_utils import get_local_cache_path, resolve_pretrained_path


__all__ = [
    "Model",
    "ModelConfig",
]

logger = get_logger(__name__)


class Model(nn.Module):
    """
    A base model for all models in Hezar.

    Args:
        config: A dataclass model config
    """

    model_filename = DEFAULT_MODEL_FILE
    config_filename = DEFAULT_MODEL_CONFIG_FILE

    def __init__(self, config: ModelConfig, *args, **kwargs):
        super().__init__()
        self.config = config.update(kwargs)

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        load_locally=False,
        model_filename=None,
        config_filename=None,
        save_path=None,
        **kwargs,
    ):
        """
        Load the model from local path or hub.

        It's recommended to actually use this method with :class:`hezar.Model` rather than any other model class
        unless you actually know that the class is the same as the one on the Hub, because the output will always be
        the one specified on the Hub!

        Args:
            hub_or_local_path: path to the model living on the Hub or local disk.
            model_filename: optional model filename.
            config_filename: optional config filename
            load_locally: force loading from local path
            save_path: save model to this path after loading

        Returns:
            The fully loaded Hezar model
        """
        # Get device if provided in the kwargs
        device = None or kwargs.pop("device", None)
        # Configure the right path
        hub_or_local_path = resolve_pretrained_path(hub_or_local_path)
        # Load config
        config_filename = config_filename or cls.config_filename
        config = ModelConfig.load(hub_or_local_path=hub_or_local_path, filename=config_filename)
        # Build model wih config
        model = build_model(config.name, config, **kwargs)
        # Raise a warning if model class is not compatible with the one on the Hub
        if cls.__name__ != "Model" and cls.__name__ != model.__class__.__name__:
            logger.warning(
                f"You attempted to load a Hub model using `{cls.__name__}` "
                f"but the model in `{hub_or_local_path}` is of type `{model.__class__.__name__}`, "
                f"So the output model is going to be a `{model.__class__.__name__}` instance!"
            )
        model_filename = model_filename or model.model_filename or cls.model_filename
        # does the path exist locally?
        is_local = load_locally or os.path.isdir(hub_or_local_path)
        if not is_local:
            model_path = hf_hub_download(
                hub_or_local_path,
                filename=model_filename,
                cache_dir=HEZAR_CACHE_DIR,
                resume_download=True,
            )
        else:
            model_path = os.path.join(hub_or_local_path, model_filename)
        # Get state dict from the model
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        if device:
            model.to(device)
        if save_path:
            model.save(save_path)
        return model

    def load_state_dict(self, state_dict, **kwargs):
        """
        Flexibly load a state dict to the model. Any incompatible or missing key is ignored and other layer weights are
        loaded. In that case a warning with additional info is raised.

        Args:
            state_dict: model state dict
            **kwargs:
        """
        incompatible_keys = []
        compatible_state_dict = OrderedDict()
        src_state_dict = self.state_dict()
        for (src_weight_key, src_weight), (trg_key, trg_weight) in zip(src_state_dict.items(), state_dict.items()):
            if src_weight.shape != trg_weight.shape:
                incompatible_keys.append(src_weight_key)
            else:
                compatible_state_dict[src_weight_key] = trg_weight

        missing_keys = []
        diff = len(src_state_dict) - len(state_dict)
        if diff > 0:
            missing_keys.extend(list(src_state_dict.keys())[-diff:])

        try:
            super().load_state_dict(compatible_state_dict, strict=True)
        except RuntimeError:
            super().load_state_dict(compatible_state_dict, strict=False)
            logger.warning(
                "Partially loading the weights as the model architecture and the given state dict are "
                "incompatible! \nIgnore this warning in case you plan on fine-tuning this model\n"
                f"Incompatible keys: {incompatible_keys}\n"
                f"Missing keys: {missing_keys}\n"
            )

    def save(
        self,
        path: Union[str, os.PathLike],
        filename: str = None,
        save_config: bool = True,
        config_filename: str = None,
    ):
        """
        Save model weights and config to a local path

        Args:
            path: A local directory to save model, config, etc.
            save_config: Whether to save config along with the model or not.
            config_filename: Model config filename,
            filename: Model weights filename

        Returns:
            Path to the saved model
        """
        # save model and config to the repo
        config_filename = config_filename or self.config_filename
        filename = filename or self.model_filename
        os.makedirs(path, exist_ok=True)
        model_save_path = os.path.join(path, filename)
        torch.save(self.state_dict(), model_save_path)
        if save_config:
            self.config.save(save_dir=path, filename=config_filename)
        return model_save_path

    def push_to_hub(self, repo_id, commit_message=None, private=False):
        """
        Push the model and required files to the hub

        Args:
            repo_id: The path (id or repo name) on the hub
            commit_message (str): Commit message for this push
            private (bool): Whether to create a private repo or not
        """
        api = HfApi()
        # create remote repo
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        # create local repo
        cache_path = get_local_cache_path(repo_id, repo_type="model")
        model_save_path = self.save(cache_path, save_config=False)
        if commit_message is None:
            commit_message = "Hezar: Upload model and config"
        # upload config file
        self.config.push_to_hub(
            repo_id, filename=self.config_filename, repo_type="model", commit_message=commit_message
        )
        # upload model file
        api.upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo=self.model_filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(
            f"Uploaded: "
            f"`{self.__class__.__name__}(name={self.config.name})`"
            f" --> "
            f"`{os.path.join(repo_id, self.model_filename)}`"
        )

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
            **kwargs: extra arguments specific to the derived class

        Returns:
            Processed model output values and converted to human-readable results
        """
        return inputs

    @torch.inference_mode()
    def predict(self, inputs, post_process=True, **kwargs) -> Dict:
        """
        Perform an end-to-end prediction on raw inputs.

        Args:
            inputs: Raw inputs e.g, a list of texts, path to images, etc.
            post_process: Whether to do post-processing step

        Returns:
            Output dict of results
        """
        self.eval()
        model_outputs = self(inputs, **kwargs)
        if post_process and hasattr(self, "post_process"):
            processed_outputs = self.post_process(model_outputs, **kwargs)
            return processed_outputs

        return model_outputs
