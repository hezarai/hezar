"""
Configs are at the core of Hezar. All core modules like `Model`, `Preprocessor`, `Trainer`, etc. take their parameters
as a config container which is an instance of `Config` or its derivatives. A `Config` is a Python dataclass with
auxiliary methods for loading, saving, uploading to the hub and etc.

Examples:
    >>> from hezar import ModelConfig
    >>> config = ModelConfig.load("hezarai/bert-base-fa")

    >>> from hezar import BertLMConfig
    >>> bert_config = BertLMConfig(vocab_size=50000, hidden_size=768)
    >>> bert_config.save("saved/bert")
    >>> bert_config.push_to_hub("hezarai/bert-custom")
"""
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from huggingface_hub import create_repo, hf_hub_download, upload_file
from omegaconf import DictConfig, OmegaConf

from .constants import DEFAULT_MODEL_CONFIG_FILE, HEZAR_CACHE_DIR
from .utils import get_logger, get_module_config_class


__all__ = [
    "Config",
    "ModelConfig",
    "PreprocessorConfig",
    "TrainConfig",
    "DatasetConfig",
    "EmbeddingConfig",
    "CriterionConfig",
    "OptimizerConfig",
    "LRSchedulerConfig",
]

logger = get_logger(__name__)


@dataclass
class Config:
    """
    Base class for all configs in Hezar.

    All configs are simple dataclasses that have some customized functionalities to manage their attributes. There are
    also some Hezar specific methods: `load`, `save` and `push_to_hub`.

    Args:
        name: A mandatory attribute that specifies a unique and pre-defined name for that config that is also used in
            the registries.
        config_type: A mandatory attribute that specifies the type of the config e.g. model, dataset, preprocessor, etc.
    """
    name: str = field(metadata={"help": "The key in the registry of that module"})
    config_type: str = "base"

    def __getitem__(self, item):
        try:
            return self.dict()[item]
        except KeyError:
            raise AttributeError(f"`{self.__class__.__name__}` has no attribute `{item}`!")

    def __len__(self):
        return len(self.dict())

    def __iter__(self):
        return iter(self.dict())

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
            if k not in self.__annotations__.keys():
                logger.warning(f"`{str(self.__class__.__name__)}` does not take `{k}` as a config parameter!")
            setattr(self, k, v)
        return self

    @classmethod
    def load(
        cls,
        hub_or_local_path: Union[str, os.PathLike],
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        """
        Load config from Hub or locally if it already exists on disk (handled by HfApi)

        Args:
            hub_or_local_path: Local or Hub path for the config
            filename: Configuration filename
            subfolder: Optional subfolder path where the config is in
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
                cache_dir=HEZAR_CACHE_DIR,
            )

        dict_config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(dict_config)
        config_cls = get_module_config_class(config["name"], config_type=config["config_type"])
        config = config_cls.from_dict(config, strict=False, **kwargs)
        return config

    @classmethod
    def from_dict(cls, dict_config: Union[Dict, DictConfig], **kwargs):
        """
        Load config from a dict-like object
        """
        # Update config parameters with kwargs
        dict_config.update(**kwargs)

        config = cls(**{k: v for k, v in dict_config.items() if hasattr(cls, k)})

        return config

    def save(self, save_dir: Union[str, os.PathLike], filename: str, subfolder: Optional[str] = None):
        """
        Save the *config.yaml file to a local path

        Args:
             save_dir: Save directory path
             filename: Config file name
             subfolder: Subfolder to save the config file
        """
        subfolder = subfolder or ""
        config = self.dict()
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
            private (bool): Whether the repo type should be private or not (ignored if the repo exists)
            commit_message (str): Push commit message
        """
        path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
        subfolder = subfolder or ""

        # create remote repo
        create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        # save to tmp and prepare for push
        cache_path = tempfile.mkdtemp()
        config_path = self.save(cache_path, filename=filename, subfolder=subfolder)
        # push to hub
        if commit_message is None:
            commit_message = f"Hezar: Upload {filename}"
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(
            f"Uploaded:`{self.__class__.__name__}(name={self.name})` --> `{os.path.join(repo_id, subfolder, filename)}`"
        )


@dataclass
class ModelConfig(Config):
    """
    Base dataclass for all model configs
    """
    name: str = field(default=None, metadata={"help": "The model's key in the models_registry"})
    config_type: str = "model"


@dataclass
class PreprocessorConfig(Config):
    """
    Base dataclass for all preprocessor configs
    """
    name: str = field(default=None, metadata={"help": "The preprocessor's key in the preprocessor_registry"})
    config_type: str = "preprocessor"


@dataclass
class DatasetConfig(Config):
    """
    Base dataclass for all dataset configs
    """
    name: str = field(default=None, metadata={"help": "The dataset's key in the datasets_registry"})
    config_type: str = "dataset"
    task: Union[str, List[str]] = field(
        default=None, metadata={"help": "Name of the task(s) this dataset is built for"}
    )


@dataclass
class EmbeddingConfig(Config):
    """
    Base dataclass for all embedding configs
    """
    name: str = field(default=None, metadata={"help": "The embedding's key in the embeddings_registry"})
    config_type: str = "embedding"


@dataclass
class CriterionConfig(Config):
    """
    Base dataclass for all criterion configs
    """
    name: str = field(default=None, metadata={"help": "The criterion's key in the criterions_registry"})
    config_type: str = "criterion"
    weight: Optional[torch.Tensor] = None
    reduce: str = None
    ignore_index: int = -100


@dataclass
class LRSchedulerConfig(Config):
    """
    Base dataclass for all scheduler configs
    """
    name: str = field(default=None, metadata={"help": "The LR scheduler's key in the schedulers_registry"})
    config_type: str = "lr_scheduler"
    verbose: bool = True


@dataclass
class OptimizerConfig(Config):
    """
    Base dataclass for all optimizer configs
    """
    name: str = field(default=None, metadata={"help": "The optimizer's key in the optimizers_registry"})
    config_type: str = "optimizer"
    lr: float = None
    weight_decay: float = .0
    scheduler: Union[Dict[str, Any], LRSchedulerConfig] = None


@dataclass
class TrainConfig(Config):
    """
    Base dataclass for all trainer configs
    """
    name: str = field(default=None, metadata={"help": "The trainer's key in the trainers_registry"})
    config_type: str = "train"
    device: str = "cuda"
    init_weights_from: str = None
    seed: int = 42
    optimizer: Union[Dict[str, Any], OptimizerConfig] = None
    batch_size: int = None
    use_amp: bool = False
    metrics: Dict[str, Dict] = field(default_factory=dict)
    num_epochs: int = None
    save_freq: int = 1
    checkpoints_dir: str = None
    log_dir: str = None
