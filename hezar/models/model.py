"""
Hezar models inherit the base class `Model`. A `Model` itself is a PyTorch Module to implement neural networks but has
some extra Hezar-specific functionalities and methods e.g, pushing to hub, loading from hub, etc.

Examples:
    >>> # Load from hub
    >>> from hezar import Model
    >>> model = Model.load("hezarai/bert-base-fa")
"""
import inspect
import os
import tempfile
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import torch
from huggingface_hub import create_repo, hf_hub_download, upload_file
from torch import nn

from ..builders import build_model
from ..configs import ModelConfig
from ..constants import (
    DEFAULT_MODEL_CONFIG_FILE,
    DEFAULT_MODEL_FILE,
    HEZAR_CACHE_DIR,
    Backends,
    LossType,
)
from ..preprocessors import Preprocessor, PreprocessorsContainer
from ..utils import Logger, verify_dependencies
from .model_outputs import ModelOutput


logger = Logger(__name__)

criterions_mapping = {
    "l1": nn.L1Loss,
    "nll": nn.NLLLoss,
    "nll_2d": nn.NLLLoss2d,
    "poisson_nll": nn.PoissonNLLLoss,
    "gaussian_nll": nn.GaussianNLLLoss,
    "mse": nn.MSELoss,
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "triple_margin": nn.TripletMarginLoss,
    "ctc": nn.CTCLoss
}


class Model(nn.Module):
    """
    Base class for all neural network models in Hezar.

    Args:
        config: A dataclass model config
    """

    required_backends: List[Union[Backends, str]] = []
    # Default file names
    model_filename = DEFAULT_MODEL_FILE
    config_filename = DEFAULT_MODEL_CONFIG_FILE

    # Specify if the model is a generative model. If True, the model must also implement the `generate` method
    is_generative: bool = False

    # Keys to ignore on loading state dicts
    skip_keys_on_load = []

    # Loss function name
    loss_fn_name: Union[str, LossType] = LossType.CROSS_ENTROPY

    def __init__(self, config: ModelConfig, *args, **kwargs):
        verify_dependencies(self, self.required_backends)
        super().__init__()
        self.config = config.update(kwargs)
        self._preprocessor = None
        self._criterion = self._set_criterion(self.loss_fn_name)

    @staticmethod
    def _set_criterion(criterion_name: str):
        if criterion_name not in criterions_mapping:
            raise ValueError(f"Invalid criterion name `{criterion_name}`. Available: {list(criterions_mapping.keys())}")
        loss_fn = criterions_mapping[criterion_name]()
        return loss_fn

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

            incompatible_keys = []

            for (src_key, src_weight), (trg_key, trg_weight) in zip(src_state_dict.items(), state_dict.items()):
                if src_weight.shape == trg_weight.shape:
                    compatible_state_dict[src_key] = trg_weight
                else:
                    # put the source key and weight if trg weight is incompatible
                    compatible_state_dict[src_key] = src_weight
                    incompatible_keys.append(src_key)

            missing_keys, _ = super().load_state_dict(compatible_state_dict, strict=False)
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
        self.save(cache_path, filename=filename, config_filename=config_filename)
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
        weights_path = os.path.join(cache_path, filename)
        upload_file(
            path_or_fileobj=weights_path,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )

        logger.log_upload_success(
            name=f"{self.__class__.__name__}(name={self.config.name})",
            target_path=os.path.join(repo_id, filename),
        )

    def forward(self, *model_inputs, **kwargs) -> Dict:
        """
        Forward inputs through the model and return logits, etc.

        Args:
            model_inputs: The required inputs for the model forward

        Returns:
            A dict of outputs like logits, loss, etc.
        """
        raise NotImplementedError

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss on the model outputs against the given labels

        Args:
            inputs: Input tensor to compute loss on
            targets: Target tensor

        Returns:
            Loss tensor
        """
        raise NotImplementedError

    def generate(self, *model_inputs, **kwargs) -> torch.Tensor:
        """
        Generation method for all generative models. Generative models have the `is_generative` attribute set to True.
        The behavior of this method is usually controlled by `generation` part of the model's config.

        Args:
            model_inputs: Model inputs for generation, usually the same as forward's `model_inputs`
            **kwargs: Generation kwargs

        Returns:
            Generated output tensor
        """
        raise NotImplementedError

    def preprocess(self, *raw_inputs: Union[Any, List[Any]], **kwargs):
        """
        Given raw inputs, preprocess the inputs and prepare them for model's `forward()`.

        Args:
            raw_inputs: Raw model inputs
            **kwargs: Extra kwargs specific to the model. See the model's specific class for more info

        Returns:
            A dict of inputs for model forward
        """
        return raw_inputs

    def post_process(self, *model_outputs: Union[Dict[str, torch.Tensor], torch.Tensor], **kwargs):
        """
        Process model outputs and return human-readable results. Called in `self.predict()`

        Args:
            model_outputs: model outputs to process
            **kwargs: extra arguments specific to the derived class

        Returns:
            Processed model output values and converted to human-readable results
        """
        return model_outputs

    @torch.inference_mode()
    def predict(
        self,
        inputs: Union[Any, List[Any]],
        device: Union[str, torch.device] = None,
        unpack_forward_inputs: bool = True,
        **kwargs,
    ) -> Union[Dict, List[Dict], torch.Tensor, Iterable, ModelOutput]:
        """
        Perform an end-to-end prediction on raw inputs.

        If the model is a generative model, it has to implement the `generate()` method too which will be called
        instead of `forward()`. (`forward()` method is called internally within the `generate()` method)

        Args:
            inputs: Raw inputs e.g, a list of texts, path to images, etc.
            device: What device to perform inference on
            unpack_forward_inputs: Whether to unpack forward inputs. Set to False if you want to send preprocess outputs
             directly to the forward/generate method without unpacking it. Note that this only applies to the cases that
             the preprocess method's output is a dict-like/mapping object.
            **kwargs: Other arguments for `preprocess`, `forward`, `generate` and `post_process`. each will be passed to
             the correct method automatically.

        Returns:
            Output dict of results
        """
        # Unpack kwargs for each step
        preprocess_kwargs, forward_kwargs, post_process_kwargs = self._unpack_prediction_kwargs(**kwargs)
        invalid_kwargs = {
            k: v for k, v in kwargs.items() if k not in {**preprocess_kwargs, **forward_kwargs, **post_process_kwargs}
        }
        if len(invalid_kwargs):
            logger.warning(
                f"Unrecognized arguments {list(invalid_kwargs.keys())} passed to `predict` method for "
                f"`{self.__class__.__name__}`"
            )

        # Put model in eval mode
        self.eval()

        # Preprocessing step
        model_inputs = self.preprocess(inputs, **preprocess_kwargs)

        # Map inputs and model to device
        device = device or self.device
        model_inputs = self._move_inputs_to_device(model_inputs, device)
        self.to(device)

        # Specify model inference function
        inference_fn = self.generate if self.is_generative else self.__call__

        # Model inference step (forward for regular models and generate for generative models)
        if isinstance(model_inputs, Mapping) and unpack_forward_inputs:
            model_outputs = inference_fn(**model_inputs, **forward_kwargs)
        else:
            model_outputs = inference_fn(model_inputs, **forward_kwargs)

        # Post-processing step
        processed_outputs = self.post_process(model_outputs, **post_process_kwargs)
        return processed_outputs

    @staticmethod
    def _move_inputs_to_device(inputs, device):
        """
        Move all input tensors in the inputs to the device

        Args:
            inputs: A torch.Tensor or a batch dict that contains tensors in its values
            device: A torch compatible device

        Returns:
            Same inputs moved to the device
        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        elif isinstance(inputs, Mapping):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            raise ValueError(
                f"Cannot move inputs of type `{type(inputs)}` to `{device}`. "
                f"Inputs data type must be either `torch.Tensor` or a mapping object like `dict`!"
            )
        return inputs

    def _unpack_prediction_kwargs(self, **kwargs):
        """
        The `predict` method can accept extra parameters for each of the `preprocess`, `forward/generate`
        and `post_process` methods. These parameters are passed as keyword arguments so that we have to make sure that
        each of them are passed to the correct method.

        Args:
            **kwargs: The kwargs to be unpacked

        Returns:
             A 3-sized tuple of (preprocess_kwargs, forward_kwargs, post_process_kwargs)
        """
        # Whether to use forward or generate based on model type
        inference_fn = type(self).generate if self.is_generative else type(self).forward

        def _get_positional_kwargs(fn):
            params = dict(inspect.signature(fn).parameters)
            params = {k: v for k, v in params.items() if v.default != v.empty}
            return params

        # Get keyword arguments from the child class (ignore positional arguments)
        preprocess_kwargs_keys = list(_get_positional_kwargs(type(self).preprocess).keys())
        post_process_kwargs_keys = list(_get_positional_kwargs(type(self).post_process).keys())
        forward_kwargs_keys = list(_get_positional_kwargs(inference_fn).keys())

        preprocess_kwargs = {k: kwargs.get(k) for k in preprocess_kwargs_keys if k in kwargs}
        forward_kwargs = {k: kwargs.get(k) for k in forward_kwargs_keys if k in kwargs}
        post_process_kwargs = {k: kwargs.get(k) for k in post_process_kwargs_keys if k in kwargs}

        return preprocess_kwargs, forward_kwargs, post_process_kwargs

    @property
    def device(self):
        """
        Get the model's device. This method is only safe when all weights of the model are on the same device.
        """
        return next(self.parameters()).device

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, value):
        if isinstance(value, str):
            self._criterion = self._set_criterion(value)
        elif isinstance(value, nn.Module):
            self._criterion = value
        else:
            raise ValueError(f"Criterion value must be either a name or a PyTorch `nn.Module`, got {type(value)}!")

    @property
    def preprocessor(self) -> PreprocessorsContainer:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Union[Preprocessor, PreprocessorsContainer, List[Preprocessor]]):
        """
        A safe setter method for model's preprocessor. Value must be either a Preprocessor, a list of Preprocessors or
        a PreprocessorsContainer instance.
        """
        if isinstance(value, Preprocessor):
            preprocessor = PreprocessorsContainer()
            preprocessor[value.config.name] = value
        elif isinstance(value, Mapping):
            preprocessor = PreprocessorsContainer(**value)
        elif isinstance(value, list):
            preprocessor = PreprocessorsContainer(**{p.config.name: p for p in value})
        elif value is None:
            preprocessor = None
        else:
            raise ValueError(
                f"Preprocessor value must be a `Preprocessor` "
                f"or a list of Preprocessor objects"
                f"or `PreprocessorContainer` instance not `{type(value)}`!"
            )
        self._preprocessor = preprocessor
