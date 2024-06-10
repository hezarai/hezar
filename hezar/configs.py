"""
Configs are at the core of Hezar. All core modules like `Model`, `Preprocessor`, `Trainer`, etc. take their parameters
as a config container which is an instance of `Config` or its derivatives. A `Config` is a Python dataclass with
auxiliary methods for loading, saving, uploading to the hub, etc.

Examples:
    >>> from hezar.configs import ModelConfig
    >>> config = ModelConfig.load("hezarai/bert-base-fa")

    >>> from hezar.models import BertMaskFillingConfig
    >>> bert_config = BertMaskFillingConfig(vocab_size=50000, hidden_size=768)
    >>> bert_config.save("saved/bert", filename="model_config.yaml")
    >>> bert_config.push_to_hub("hezarai/bert-custom", filename="model_config.yaml")
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pprint import pformat
from typing import Any, Dict, List, Literal, Optional, Tuple

from huggingface_hub import create_repo, hf_hub_download, upload_file
from omegaconf import DictConfig, OmegaConf

from .constants import (
    DEFAULT_MODEL_CONFIG_FILE,
    HEZAR_CACHE_DIR,
    ConfigType,
    LRSchedulerType,
    OptimizerType,
    PrecisionType,
    TaskType,
)
from .utils import Logger, get_module_config_class


__all__ = [
    "Config",
    "ModelConfig",
    "PreprocessorConfig",
    "TrainerConfig",
    "DatasetConfig",
    "EmbeddingConfig",
    "MetricConfig",
]

logger = Logger(__name__)

CONFIG_CLASS_VARS = ["name", "config_type"]

_config_to_type_mapping = {
    "ModelConfig": ConfigType.MODEL,
    "PreprocessorConfig": ConfigType.PREPROCESSOR,
    "TrainerConfig": ConfigType.TRAINER,
    "DatasetConfig": ConfigType.DATASET,
    "EmbeddingConfig": ConfigType.EMBEDDING,
    "CriterionConfig": ConfigType.CRITERION,
    "MetricConfig": ConfigType.METRIC,
}
_type_to_config_mapping = {v: k for k, v in _config_to_type_mapping.items()}


@dataclass
class Config:
    """
    Base class for all configs in Hezar.

    All configs are simple dataclasses that have some customized functionalities to manage their attributes. There are
    also some Hezar specific methods: `load`, `save` and `push_to_hub`.

    """

    name: str = field(init=False, default=None)
    config_type: str = field(init=False, default=ConfigType.BASE)

    def __post_init__(self):
        # Class variables cannot be init-able
        fields_dict = {f.name: f for f in fields(self)}
        for attr in CONFIG_CLASS_VARS:
            if fields_dict[attr].init == True:  # noqa
                raise ValueError(
                    f"The parameter `{attr}` in a config should be either non-initable or unannotated! "
                    f"\nYou should define it as either:\n"
                    f"`{attr} = '{getattr(self, attr)}'`"
                    f" or "
                    f"`{attr}: str = field(default='{getattr(self, attr)}', init=False)`"
                )

        # Convert enums to values
        for param, value in self.dict().items():
            if isinstance(getattr(self, param), Enum):
                setattr(self, param, str(getattr(self, param)))

    def __str__(self):
        return pformat(self.dict())

    def __getitem__(self, item):
        try:
            return self.dict()[item]
        except KeyError:
            raise AttributeError(f"`{self.__class__.__name__}` does not have the parameter `{item}`!")

    def __len__(self):
        return len(self.dict())

    def __iter__(self):
        return iter(self.dict())

    @classmethod
    def fields(cls):
        return cls.__dataclass_fields__

    def dict(self):
        """
        Returns the config object as a dictionary (works on nested dataclasses too)

        Returns:
            The config object as a dictionary
        """
        return asdict(self)

    def keys(self):
        return list(self.dict().keys())

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, d: dict, **kwargs):
        """
        Update config with a given dictionary or keyword arguments. If a key does not exist in the attributes, prints a
        warning but sets it anyway.

        Args:
            d: A dictionary
            **kwargs: Key/value pairs in the form of keyword arguments

        Returns:
            The config object itself but the operation happens in-place anyway
        """
        d.update(kwargs)
        for k, v in d.items():
            if k not in self.fields():
                logger.warning(f"`{str(self.__class__.__name__)}` does not take `{k}` as a config parameter!")
            setattr(self, k, v)
        return self

    @classmethod
    def load(
        cls,
        hub_or_local_path: str | os.PathLike,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        repo_type: str = None,
        cache_dir: str = None,
        **kwargs,
    ) -> "Config":
        """
        Load config from Hub or locally if it already exists on disk (handled by HfApi)

        Args:
            hub_or_local_path: Local or Hub path for the config
            filename: Configuration filename
            subfolder: Optional subfolder path where the config is in
            repo_type: Repo type e.g, model, dataset, etc
            cache_dir: Path to cache directory
            **kwargs: Manual config parameters to override

        Returns:
            A Config instance
        """
        filename = filename or DEFAULT_MODEL_CONFIG_FILE
        subfolder = subfolder or ""

        config_path = os.path.join(hub_or_local_path, subfolder, filename)
        is_local = os.path.isfile(config_path)
        if os.path.isdir(hub_or_local_path) and not is_local:
            raise EnvironmentError(
                f"Path `{hub_or_local_path}` exists locally but the config file {filename} is missing!"
            )
        # if the file or repo_id does not exist locally, load from the Hub
        if not is_local:
            config_path = hf_hub_download(
                hub_or_local_path,
                filename=filename,
                subfolder=subfolder,
                cache_dir=cache_dir or HEZAR_CACHE_DIR,
                repo_type=repo_type,
            )
        # Load config file and convert to dictionary
        dict_config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(dict_config)
        # Check if config_type in the file and class are equal
        config_type = config.get("config_type", ConfigType.BASE)
        if config_type in _config_to_type_mapping.values():
            if config_type != cls.config_type:
                raise ValueError(
                    f"The `config_type` for `{cls.__name__}` is `{cls.config_type}` "
                    f"which is different from the `config_type` parameter in `{filename}` which is `{config_type}`!"
                )
        config_cls = get_module_config_class(config["name"], registry_type=config_type)
        if config_cls is None:
            config_cls = cls
        config = config_cls.from_dict(config, **kwargs)
        return config

    @classmethod
    def from_dict(cls, dict_config: Dict | DictConfig, **kwargs):
        """
        Load config from a dict-like object. Nested configs are also recursively converted to their classes if possible.
        """
        # Update config parameters with kwargs
        dict_config.update(**kwargs)

        for k, v in dict_config.items():
            if isinstance(v, Dict) and "name" in v and "config_type" in v:
                config_cls = get_module_config_class(v["name"], v["config_type"])
                if config_cls is not None:
                    dict_config[k] = config_cls.from_dict(v)

        dict_config = {k: v for k, v in dict_config.items() if k in cls.fields() and cls.fields()[k].init}

        config = cls(**dict_config)  # noqa

        return config

    def save(
        self,
        save_dir: str | os.PathLike,
        filename: str,
        subfolder: Optional[str] = None,
        skip_none_fields: Optional[bool] = True,
    ):
        """
        Save the `*config.yaml` file to a local path

        Args:
             save_dir: Save directory path
             filename: Config file name
             subfolder: Subfolder to save the config file
             skip_none_fields (bool): Whether to skip saving None values or not
        """
        subfolder = subfolder or ""
        config = self.dict()

        if skip_none_fields:
            # exclude None items
            config = {k: v for k, v in config.items() if v is not None}

        # make and save to directory
        os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)
        save_path = os.path.join(save_dir, subfolder, filename)
        OmegaConf.save(config, save_path)

        return save_path

    def push_to_hub(
        self,
        repo_id: str,
        filename: str,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = "model",
        skip_none_fields: Optional[bool] = True,
        private: Optional[bool] = False,
        commit_message: Optional[str] = None,
    ):
        """
        Push the config file to the hub

        Args:
            repo_id (str): Repo name or id on the Hub
            filename (str): config file name
            subfolder (str): subfolder to save the config
            repo_type (str): Type of the repo e.g, model, dataset, space
            skip_none_fields (bool): Whether to skip saving None values or not
            private (bool): Whether the repo type should be private or not (ignored if the repo exists)
            commit_message (str): Push commit message
        """
        path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
        subfolder = subfolder or ""

        # create remote repo
        create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        # save to tmp and prepare for push
        cache_path = tempfile.mkdtemp()
        config_path = self.save(cache_path, filename=filename, subfolder=subfolder, skip_none_fields=skip_none_fields)
        # push to hub
        if commit_message is None:
            commit_message = f"Hezar: Upload {filename}"
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        logger.log_upload_success(name=f"{self.__class__.__name__}()", target_path=os.path.join(repo_id, path_in_repo))


@dataclass
class ModelConfig(Config):
    """
    Base dataclass for all model configs
    """

    name: str = field(init=False, default=None)
    config_type: str = field(init=False, default=ConfigType.MODEL)


@dataclass
class PreprocessorConfig(Config):
    """
    Base dataclass for all preprocessor configs
    """

    name: str = field(init=False, default=None)
    config_type: str = field(init=False, default=ConfigType.PREPROCESSOR)


@dataclass
class DatasetConfig(Config):
    """
    Base dataclass for all dataset configs
    """

    name: str = field(init=False, default=None)
    config_type: str = field(init=False, default=ConfigType.DATASET)
    task: TaskType | List[TaskType] = field(
        default=None, metadata={"help": "Name of the task(s) this dataset is built for"}
    )
    path: str = None


@dataclass
class EmbeddingConfig(Config):
    """
    Base dataclass for all embedding configs
    """

    name: str = field(init=False, default=None)
    config_type: str = field(init=False, default=ConfigType.EMBEDDING)
    bypass_version_check: bool = field(
        default=False,
        metadata={"help": "Whether to bypass checking gensim/numpy/hezar version compatibility"},
    )


@dataclass
class MetricConfig(Config):
    """
    Base dataclass config for all metric configs
    """

    name: str = field(init=False, default=None)
    config_type: str = field(init=False, default=ConfigType.METRIC)
    objective: Literal["maximize", "minimize"] = None
    output_keys: List | Tuple = None
    n_decimals: int = 4


@dataclass
class TrainerConfig(Config):
    """
    Base dataclass for all trainer configs

    Args:
        task (str, TaskType):
            The training task. Must be a valid name from `TaskType`.
        output_dir (str):
            Path to the directory to save trainer properties.
        device (str):
            Hardware device e.g, `cuda:0`, `cpu`, etc.
        num_epochs (int):
            Number of total epochs to train the model.
        init_weights_from (str):
            Path to a model from disk or Hub to load the initial weights from. Note that this only loads the model
            weights and ignores other checkpoint-related states if the path is a checkpoint. To resume training from
            a checkpoint use the `resume` parameter.
        resume_from_checkpoint (bool, str, os.PathLike):
            Resume training from a checkpoint. If set to True, the trainer will load the latest checkpoint, otherwise if
            a path to a checkpoint is given, it will load that checkpoint and all the other states corresponding to that
            checkpoint.
        max_steps (int):
            Maximum number of iterations to train. This helps to limit how many batches you want to train in total.
        num_dataloader_workers (int):
            Number of dataloader workers, defaults to 4 .
        dataloader_shuffle (bool):
            Control dataloaders `shuffle` argument.
        seed (int):
            Control determinism of the run by setting a seed value. defaults to 42.
        optimizer (OptimizerType):
            Name of the optimizer, available values include properties in `OptimizerType` enum.
        learning_rate (float):
            Initial learning rate for the optimizer.
        weight_decay (float):
            Optimizer weight decay value.
        lr_scheduler (LRSchedulerType):
            Optional learning rate scheduler among `LRSchedulerType` enum.
        lr_scheduler_kwargs (Dict[str, Any]):
            LR scheduler instructor kwargs depending on the scheduler type
        batch_size (int):
            Training batch size.
        eval_batch_size (int):
            Evaluation batch size, defaults to `batch_size` if None.
        gradient_accumulation_steps (int):
            Number of updates steps to accumulate before performing a backward/update pass, defaults to 1.
        distributed (bool):
            Whether to use distributed training (via the `accelerate` package)
        mixed_precision (PrecisionType | str):
            Mixed precision type e.g, fp16, bf16, etc. (disabled by default)
        use_cpu (bool):
            Whether to train using the CPU only even if CUDA is available.
        do_evaluate (bool):
            Whether to run evaluation when calling `Trainer.train`
        evaluate_with_generate (bool):
            Whether to use `generate()` in the evaluation step or not. (only applicable for generative models).
        metrics (List[str | MetricConfig]):
            A list of metrics. Depending on the `valid_metrics` in the specific MetricsHandler of the Trainer.
        metric_for_best_model (str):
            Reference metric key to watch for determining the best model. Recommended to have a {train. | evaluation.}
            prefix (e.g, evaluation.f1, train.accuracy, etc.) but if not, defaults to
            `evaluation.{metric_for_best_model}`.
        save_freq (int) (DEPRECATED):
            Deprecated and renamed to `save_steps`.
        save_steps (int):
            Save the trainer outputs every `save_steps` steps. Leave as `0` to ignore saving between training steps.
        log_steps (int):
            Save training metrics every `log_steps` steps.
        checkpoints_dir (str):
            Path to the checkpoints' folder. The actual files will be saved under `{output_dir}/{checkpoints_dir}`.
        logs_dir (str):
            Path to the logs' folder. The actual log files will be saved under `{output_dir}/{logs_dir}`.
    """

    name: str = field(init=False, default="trainer")
    config_type: str = field(init=False, default=ConfigType.TRAINER)
    output_dir: str
    task: str | TaskType
    device: str = "cuda"
    num_epochs: int = None
    init_weights_from: str = None
    resume_from_checkpoint: bool | str | os.PathLike = None
    max_steps: int = None
    num_dataloader_workers: int = 0
    dataloader_shuffle: bool = True
    seed: int = 42
    optimizer: str | OptimizerType = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    lr_scheduler: str | LRSchedulerType = None
    lr_scheduler_kwargs: Dict[str, Any] = None
    batch_size: int = None
    eval_batch_size: int = None
    gradient_accumulation_steps: int = 1
    distributed: bool = False
    mixed_precision: PrecisionType | str | None = None
    use_cpu: bool = False
    do_evaluate: bool = True
    evaluate_with_generate: bool = True
    metrics: List[str | MetricConfig] = None
    metric_for_best_model: str = "loss"
    save_enabled: bool = True
    save_freq: int = None
    save_steps: int = None
    log_steps: int = None
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"

    def __post_init__(self):
        """
        Perform some argument sanitization and filtering here to avoid unexpected behavior in the trainer.
        The need for having this method is that some fields in the Trainer's config have correlations with each other
        and not controlling them can lead to conflicts.
        """
        super().__post_init__()
        # Validate `task`
        if self.task not in list(TaskType):
            raise ValueError(
                f"Invalid task `{self.task}` passed to `TrainerConfig`. "
                f"Available options are {TaskType.list()}",
            )
        # Validate `metric_for_best_model`
        if not (self.metric_for_best_model.startswith("evaluation") or self.metric_for_best_model.startswith("train")):
            if self.do_evaluate:
                self.metric_for_best_model = f"evaluation.{self.metric_for_best_model}"
            else:
                self.metric_for_best_model = f"train.{self.metric_for_best_model}"

        # Validate steps
        if self.save_steps is not None and self.save_steps % self.gradient_accumulation_steps != 0:
            logger.warning(
                f"It's recommended to set a `save_steps` dividable by `gradient_accumulation_steps`, "
                f"otherwise, the saved model will have non-updated weights!\n"
                f"`save_steps={self.save_steps}`, `gradient_accumulation_steps={self.gradient_accumulation_steps}`"
            )

        # Validate deprecated fields
        if self.save_freq is not None:
            logger.warning(
                "Trainer argument `save_freq` is deprecated! Use `save_steps` (number of training steps per save)."
                "Note that saving is also done at the end of each epoch unless you set `save_enabled` to `False` !"
            )

        # Distributed mode
        if self.distributed:
            logger.warning(
                "Distributed mode is experimental and might have bugs. Use with caution and don't forget to submit your issues!"
            )

        # Disable tokenizers parallelism if num_dataloader_workers is more than 1
        if self.num_dataloader_workers > 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
