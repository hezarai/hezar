"""
Hezar models inherit the base class `Model`. A `Model` itself is a PyTorch Module to implement neural networks but has
some extra Hezar-specific functionalities and methods e.g, pushing to hub, loading from hub, etc.

Examples:
    >>> # Load from hub
    >>> from hezar import Model
    >>> model = Model.load("hezarai/bert-base-fa")
"""
import os
import tempfile
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from huggingface_hub import create_repo, hf_hub_download, upload_file
from torch import nn

from ..builders import build_model
from ..configs import ModelConfig
from ..constants import DEFAULT_MODEL_CONFIG_FILE, DEFAULT_MODEL_FILE, HEZAR_CACHE_DIR, Backends
from ..preprocessors import Preprocessor, PreprocessorsContainer
from ..utils import Logger, verify_dependencies


__all__ = [
    "Model",
    "GenerativeModel",
]

logger = Logger(__name__)


class Model(nn.Module):
    """
    A base model for all neural-based models in Hezar.

    Args:
        config: A dataclass model config
    """
    required_backends: List[Union[Backends, str]] = []
    # Default file names
    model_filename = DEFAULT_MODEL_FILE
    config_filename = DEFAULT_MODEL_CONFIG_FILE

    # Keys to ignore on loading state dicts
    skip_keys_on_load = []

    def __init__(self, config: ModelConfig, *args, **kwargs):
        verify_dependencies(self, self.required_backends)
        super().__init__()
        self.config = config.update(kwargs)
        self._preprocessor = None

    @classmethod
    def load(
        cls,
        hub_or_local_path: Union[str, os.PathLike],
        load_locally: Optional[bool] = False,
        load_preprocessor: Optional[bool] = True,
        model_filename: Optional[str] = None,
        config_filename: Optional[str] = None,
        save_path: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> "Model":
        """
        Load the model from local path or hub.

        It's recommended to actually use this method with :class:`hezar.Model` rather than any other model class
        unless you actually know that the class is the same as the one on the Hub, because the output will always be
        the one specified on the Hub!

        Args:
            hub_or_local_path: Path to the model living on the Hub or local disk.
            load_locally: Force loading from local path
            load_preprocessor: Whether to load the preprocessor(s) or not
            model_filename: Optional model filename.
            config_filename: Optional config filename
            save_path: Save model to this path after loading

        Returns:
            The fully loaded Hezar model
        """
        # Get device if provided in the kwargs
        device = None or kwargs.pop("device", None)
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
        # Load the preprocessor(s)
        if load_preprocessor:
            preprocessor = Preprocessor.load(hub_or_local_path, force_return_dict=True)
            model.preprocessor = preprocessor
        return model

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """
        Flexibly load the state dict to the model.

        Any incompatible or missing key is ignored and other layer weights are
        loaded. In that case a warning with additional info is raised.

        Args:
            state_dict: Model state dict
        """
        if len(self.skip_keys_on_load):
            for key in self.skip_keys_on_load:
                if key in state_dict:
                    state_dict.pop(key, None)  # noqa
        try:
            super().load_state_dict(state_dict, strict=True)
        except RuntimeError:
            compatible_state_dict = OrderedDict()
            src_state_dict = self.state_dict()
            for (src_key, src_weight), (trg_key, trg_weight) in zip(src_state_dict.items(), state_dict.items()):
                if src_weight.shape == trg_weight.shape:
                    compatible_state_dict[src_key] = trg_weight
                else:
                    compatible_state_dict[trg_key] = trg_weight

            missing_keys, incompatible_keys = super().load_state_dict(compatible_state_dict, strict=False)
            if len(missing_keys) or len(incompatible_keys):
                logger.warning(
                    "Partially loading the weights as the model architecture and the given state dict are "
                    "incompatible! \nIgnore this warning in case you plan on fine-tuning this model\n"
                    f"Incompatible keys: {incompatible_keys}\n"
                    f"Missing keys: {missing_keys}\n"
                )

    def save(
        self,
        path: Union[str, os.PathLike],
        filename: Optional[str] = None,
        save_preprocessor: Optional[bool] = True,
        config_filename: Optional[str] = None,
    ):
        """
        Save model weights and config to a local path

        Args:
            path: A local directory to save model, config, etc.
            save_preprocessor: Whether to save preprocessor(s) along with the model or not
            config_filename: Model config filename,
            filename: Model weights filename

        Returns:
            Path to the saved model
        """
        # save model and config to the repo
        config_filename = config_filename or self.config_filename
        filename = filename or self.model_filename
        os.makedirs(path, exist_ok=True)

        self.config.save(save_dir=path, filename=config_filename)

        model_save_path = os.path.join(path, filename)
        torch.save(self.state_dict(), model_save_path)

        if save_preprocessor:
            if self.preprocessor is not None:
                self.preprocessor.save(path)
        return model_save_path

    def push_to_hub(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        config_filename: Optional[str] = None,
        push_preprocessor: Optional[bool] = True,
        commit_message: Optional[str] = None,
        private: Optional[bool] = False,
    ):
        """
        Push the model and required files to the hub

        Args:
            repo_id: The path (id or repo name) on the hub
            filename: Model file name
            config_filename: Config file name
            push_preprocessor: Whether to push preprocessor(s) or not
            commit_message (str): Commit message for this push
            private (bool): Whether to create a private repo or not
        """
        config_filename = config_filename or self.config_filename
        filename = filename or self.model_filename

        # create remote repo
        create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

        # save to tmp and prepare for push
        cache_path = tempfile.mkdtemp()
        model_save_path = self.save(
            cache_path,
            filename=filename,
            config_filename=config_filename,
        )
        if commit_message is None:
            commit_message = "Hezar: Upload model and config"

        # upload config file
        self.config.push_to_hub(
            repo_id,
            filename=config_filename,
            repo_type="model",
            commit_message=commit_message,
        )
        # upload preprocessor(s)
        if push_preprocessor:
            if self.preprocessor is not None:
                self.preprocessor.push_to_hub(repo_id, commit_message=commit_message, private=private)

        # upload model file
        upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )

        logger.log_upload_success(
            name=f"{self.__class__.__name__}(name={self.config.name})",
            target_path=os.path.join(repo_id, filename),
        )

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
        Given raw inputs, preprocess the inputs and prepare them for model forward.

        Args:
            inputs: Raw model inputs
            **kwargs: Extra kw arguments

        Returns:
            A dict of inputs for model forward
        """
        return inputs

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
    def predict(self, inputs, **kwargs) -> Union[Dict, List[Dict]]:
        """
        Perform an end-to-end prediction on raw inputs.

        Args:
            inputs: Raw inputs e.g, a list of texts, path to images, etc.

        Returns:
            Output dict of results
        """
        # Put model in eval mode
        self.eval()
        # Preprocessing step
        if self.preprocessor is not None:
            inputs = self.preprocess(inputs, **kwargs)
        # Model forward step
        model_outputs = self(inputs, **kwargs)
        # Post-processing step
        processed_outputs = self.post_process(model_outputs, **kwargs)
        return processed_outputs

    @property
    def device(self):
        """
        Get the model's device. This method is only safe when all weights on the model are on the same device.
        """
        return next(self.parameters()).device

    @property
    def preprocessor(self) -> PreprocessorsContainer:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Union[Preprocessor, PreprocessorsContainer]):
        if isinstance(value, Preprocessor):
            preprocessor = PreprocessorsContainer()
            preprocessor[value.config.name] = value
        elif isinstance(value, Mapping):
            preprocessor = PreprocessorsContainer(**value)
        elif value is None:
            preprocessor = None
        else:
            raise ValueError(f"Preprocessor value must be a `Preprocessor` or a `PreprocessorContainer` instance"
                             f"not `{type(value)}`!")
        self._preprocessor = preprocessor


class GenerativeModel(Model):
    """
    A Model subclass specific to generative models
    """
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    @torch.inference_mode()
    def generate(self, inputs, **kwargs):
        """
        Generation method for all generative models. The behavior of this method is usually controlled by `generation`
        part of the model config.

        Args:
            inputs: Preprocessed input ids
            **kwargs: Generation kwargs

        Returns:
            Generated ids
        """
        raise NotImplementedError(f"`{self.__class__.__name__}` is a generative model "
                                  f"but has not implemented the `generate()` method!")

    @torch.inference_mode()
    def predict(self, inputs, **kwargs):
        """
        Perform an end-to-end prediction on raw inputs designed for generative models.

        Args:
            inputs: Raw inputs e.g, a list of texts, path to images, etc.

        Returns:
            Output dict of results
        """
        # Put model in eval mode
        self.eval()
        # Preprocessing step
        if self.preprocessor is not None:
            inputs = self.preprocess(inputs, **kwargs)
        # Model forward step
        model_outputs = self.generate(inputs, **kwargs)
        # Post-processing step
        processed_outputs = self.post_process(model_outputs, **kwargs)
        return processed_outputs
