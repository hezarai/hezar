from __future__ import annotations

import math
import os
from typing import List

from torch.utils.data import Dataset as TorchDataset

from ...builders import build_dataset
from ...configs import DatasetConfig
from ...constants import (
    DEFAULT_DATASET_CONFIG_FILE,
    HEZAR_CACHE_DIR,
    Backends,
    RepoType,
    SplitType,
)
from ...preprocessors import Preprocessor, PreprocessorsContainer
from ...utils import Logger, get_module_config_class, list_repo_files, verify_dependencies


logger = Logger(__name__)


class Dataset(TorchDataset):
    """
    Base class for all datasets in Hezar.

    Args:
        config: The configuration object for the dataset.
        split: Dataset split name e.g, train, test, validation, etc.
        preprocessor: Preprocessor object or path (note that Hezar datasets classes require this argument).
        **kwargs: Additional keyword arguments.

    Attributes:
        required_backends (List[str | Backends]): List of required backends for the dataset.
        config_filename (str): Default dataset config file name.
        cache_dir (str): Default cache directory for the dataset.

    """
    required_backends: List[str | Backends] = [Backends.DATASETS]
    config_filename = DEFAULT_DATASET_CONFIG_FILE
    cache_dir = os.path.join(HEZAR_CACHE_DIR, "datasets")

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        preprocessor: str | Preprocessor | PreprocessorsContainer = None,
        **kwargs,
    ):
        verify_dependencies(self, self.required_backends)
        self.config = config.update(kwargs)
        self.split = split
        self.data = self._load(self.split)
        self.preprocessor = self.create_preprocessor(preprocessor)
        self.data_collator = None

    def _load(self, split):
        """
        The internal function to load the dataset files and properties.
        By default, this uses the HF `datasets.load_dataset()`.
        """
        pass

    @staticmethod
    def create_preprocessor(preprocessor: str | Preprocessor | PreprocessorsContainer):
        """
        Create the preprocessor for the dataset.

        Args:
            preprocessor (str | Preprocessor | PreprocessorsContainer): Preprocessor for the dataset
        """
        if preprocessor is None:
            logger.warning(
                "Since v0.39.0, `Dataset` classes require the `preprocessor` parameter and cannot be None or it will "
                "lead to errors later on! (This warning will change to an error in the future)"
            )
            return PreprocessorsContainer()

        if isinstance(preprocessor, str):
            preprocessor = Preprocessor.load(preprocessor)
        if isinstance(preprocessor, Preprocessor):
            preprocessor = PreprocessorsContainer({preprocessor.config.name: preprocessor})
        elif isinstance(preprocessor, PreprocessorsContainer):
            preprocessor = preprocessor
        else:
            raise ValueError(
                f"The `preprocessor` must be a path to the Hub or a Preprocessor/PreprocessorsContainer instance, "
                f"got {type(preprocessor)}!"
            )
        return preprocessor

    def __str__(self):
        dataset_name = self.config.path or self.config.name
        dataset_size = len(self)
        return f"{self.__class__.__name__}(path={dataset_name}['{self.split}'], size={dataset_size})"

    def __len__(self):
        """
        Returns the length of the dataset. The `max_size` parameter in the config can overwrite this value. Override
        with caution!
        """
        if isinstance(self.config.max_size, float) and 0 < self.config.max_size <= 1:
            return math.ceil(self.config.max_size * len(self.data))

        elif isinstance(self.config.max_size, int) and 0 < self.config.max_size < len(self.data):
            return self.config.max_size

        return len(self.data)

    def __getitem__(self, index):
        """
        Gets a specific item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.

        """
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        hub_path: str | os.PathLike,
        split: str | SplitType = None,
        preprocessor: str | Preprocessor | PreprocessorsContainer = None,
        config: DatasetConfig = None,
        config_filename: str = None,
        cache_dir: str = None,
        **kwargs,
    ) -> "Dataset":
        """
        Load the dataset from a hub path.

        Args:
            hub_path (str | os.PathLike):
                Path to dataset from hub or locally.
            split (Optional[str | SplitType]):
                Dataset split, defaults to "train".
            preprocessor (str | Preprocessor | PreprocessorsContainer):
                Preprocessor object for the dataset
            config: (DatasetConfig):
                A config object to ignore the config in the repo or in case the repo has no `dataset_config.yaml` file
            config_filename (Optional[str]):
                Dataset config file name. Falls back to `dataset_config.yaml` if not given.
            cache_dir (str):
                Path to cache directory, defaults to Hezar's cache directory
            **kwargs:
                Config parameters as keyword arguments.

        Returns:
            Dataset: An instance of the loaded dataset.

        """
        split = split or "train"
        config_filename = config_filename or cls.config_filename

        if ":" in hub_path:
            hub_path, hf_dataset_config_name = hub_path.split(":")
            kwargs["hf_load_kwargs"] = kwargs.get("hf_load_kwargs", {})
            kwargs["hf_load_kwargs"]["name"] = hf_dataset_config_name

        if cache_dir is not None:
            cls.cache_dir = cache_dir

        has_config = config_filename in list_repo_files(hub_path, repo_type="dataset")

        if config is not None:
            dataset_config = config.update(kwargs)
        elif has_config:
            dataset_config = DatasetConfig.load(
                hub_path,
                filename=config_filename,
                repo_type=RepoType.DATASET,
                cache_dir=cls.cache_dir,
                **kwargs,
            )
        elif kwargs.get("task", None):
            config_cls = get_module_config_class(kwargs["task"], registry_type="dataset")
            if config_cls:
                dataset_config = config_cls(**kwargs)
            else:
                raise ValueError(f"Task `{kwargs['task']}` is not valid!")
        else:
            raise ValueError(
                f"The dataset at `{hub_path}` does not have enough info and config to load using Hezar!"
                f"\nHint: Either pass the proper `config` to `.load()` or pass in required config parameters as "
                f"kwargs in `.load()`, most notably `task`!"
            )

        dataset_config.path = hub_path
        dataset = build_dataset(
            dataset_config.name,
            config=dataset_config,
            split=split,
            preprocessor=preprocessor,
            **kwargs,
        )
        return dataset
