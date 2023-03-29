import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig, OmegaConf

from .constants import HEZAR_CACHE_DIR
from .utils import get_local_cache_path, get_logger, get_module_config_class, resolve_pretrained_path


__all__ = [
    "Config",
    "ModelConfig",
    "PreprocessorConfig",
    "TrainConfig",
    "DatasetConfig",
    "CriterionConfig",
    "OptimizerConfig",
    "LRSchedulerConfig",
]

logger = get_logger(__name__)


@dataclass
class Config:
    """
    Base class for all the configs in Hezar.

    All configs are simple dataclasses that have some customized functionalities to manage their attributes. There are
    also some Hezar specific methods: `load`, `save` and `push_to_hub`.

    Args:
        name: a mandatory attribute that specifies a unique and pre-defined name for that config that is also used in
            the registries.
        config_type: a mandatory attribute that specifies the type of the config e.g. model, dataset, preprocessor, etc.
    """
    name: str
    config_type: str = "base"

    def __getitem__(self, item):
        try:
            return self.dict()[item]
        except KeyError:
            raise ValueError(f"`{self.__class__.__name__}` has no attribute `{item}`!")

    def __len__(self):
        return len(self.dict())

    def __iter__(self):
        return iter(self.dict())

    def dict(self):
        return self.__dict__

    def keys(self):
        return self.dict().keys()

    def pop(self, key, default=None):
        if key in self.__annotations__.keys():
            value = getattr(self, key)
            delattr(self, key)
        else:
            value = default
        return value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, d: dict, **kwargs):
        d.update(kwargs)
        for k, v in d.items():
            if k not in self.__annotations__.keys():
                logger.warning(f"`{str(self.__class__.__name__)}` does not take `{k}` as a config parameter!")
            setattr(self, k, v)
        return self

    @classmethod
    def load(cls, hub_or_local_path: Union[str, os.PathLike], filename="config.yaml", subfolder="", **kwargs):
        """
        Load config from Hub or locally if it already exists on disk (handled by HfApi)
        """
        hub_or_local_path = resolve_pretrained_path(hub_or_local_path)
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

    def save(self, save_dir, filename="config.yaml", subfolder=""):
        """
        Save the *config.yaml file to a local path

        Args:
             save_dir: save directory path
             filename: config file name
             subfolder: subfolder to save the config file
        """
        config = self.dict()
        # exclude None items
        config = {k: v for k, v in config.items() if v is not None}
        # make and save to directory
        os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)
        save_path = os.path.join(save_dir, subfolder, filename)
        OmegaConf.save(config, save_path)
        return save_path

    def push_to_hub(self, hub_path, filename, subfolder="", repo_type="model", private=False, commit_message=None):
        """
        Push the config file to the hub

        Args:
            hub_path (str): Repo name or id on the Hub
            filename (str): config file name
            subfolder (str): subfolder to save the config
            repo_type (str): Type of the repo e.g, model, dataset, space
            private (bool): Whether the repo type should be private or not (ignored if the repo exists)
            commit_message (str): Push commit message
        """
        api = HfApi()
        repo_id = resolve_pretrained_path(hub_path)
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        cache_path = get_local_cache_path(repo_id, repo_type=repo_type)
        config_path = self.save(cache_path, filename=filename, subfolder=subfolder)
        # push to hub
        if commit_message is None:
            commit_message = f"Hezar: Upload {filename}"
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=os.path.join(subfolder, filename),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Uploaded `{self.name}` config to `{repo_id}/{subfolder}` as `{filename}`")


@dataclass
class ModelConfig(Config):
    name: str = field(default=None, metadata={"help": "Name of the model's key in the models_registry"})
    config_type: str = "model"


@dataclass
class PreprocessorConfig(Config):
    name: str = field(default=None, metadata={"help": "Name of the preprocessor's key in the preprocessor_registry"})
    config_type: str = "preprocessor"


@dataclass
class DatasetConfig(Config):
    name: str = field(default=None, metadata={"help": "Name of the dataset"})
    config_type: str = "dataset"
    task: Union[str, List[str]] = field(
        default=None, metadata={"help": "Name of the task(s) this dataset is built for"}
    )


@dataclass
class CriterionConfig(Config):
    name: str = None
    config_type: str = "criterion"
    weight: Optional[torch.Tensor] = None
    reduce: str = None
    ignore_index: int = -100


@dataclass
class LRSchedulerConfig(Config):
    name: str = None
    config_type: str = "lr_scheduler"
    verbose: bool = True


@dataclass
class OptimizerConfig(Config):
    name: str = None
    config_type: str = "optimizer"
    lr: float = None
    weight_decay: float = None


@dataclass
class TrainConfig(Config):
    name: str = field(default=None)
    config_type: str = "train"
    device: str = "cuda"
    init_weights_from: str = None
    seed: int = 42
    optimizer: OptimizerConfig = None
    batch_size: int = field(default=None, metadata={"help": "training batch size"})
    metrics: Dict[str, Dict] = field(default_factory=list)
    num_train_epochs: int = None
    checkpoints_dir: str = None
    log_dir: str = None
