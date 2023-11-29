import os
from typing import List, Optional, Union

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
from ...utils import verify_dependencies


class Dataset(TorchDataset):
    """
    Base class for all datasets in Hezar.

    Args:
        config (DatasetConfig): The configuration object for the dataset.
        **kwargs: Additional keyword arguments.

    Attributes:
        required_backends (List[Union[str, Backends]]): List of required backends for the dataset.
        config_filename (str): Default dataset config file name.
        cache_dir (str): Default cache directory for the dataset.

    """
    required_backends: List[Union[str, Backends]] = None
    config_filename = DEFAULT_DATASET_CONFIG_FILE
    cache_dir = HEZAR_CACHE_DIR

    def __init__(self, config: DatasetConfig, split=None, **kwargs):
        """
        Initializes a new dataset instance.

        Args:
            config (DatasetConfig): The configuration object for the dataset.
            **kwargs: Additional keyword arguments.

        """
        verify_dependencies(self, self.required_backends)
        self.config = config.update(kwargs)
        self.preprocessor = None
        self.data_collator = None
        self.split = split

    def __str__(self):
        dataset_name = self.config.path or self.config.name
        dataset_size = len(self)
        return f"{self.__class__.__name__}(path={dataset_name}['{self.split}'], size={dataset_size})"

    def __len__(self):
        """
        Returns the length of the dataset.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.

        """
        raise NotImplementedError

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
        hub_path: Union[str, os.PathLike],
        config_filename: Optional[str] = None,
        split: Optional[Union[str, SplitType]] = None,
        **kwargs,
    ) -> "Dataset":
        """
        Load the dataset from a hub path.

        Args:
            hub_path (Union[str, os.PathLike]): Path to dataset from hub or locally.
            config_filename (Optional[str]): Dataset config file name.
            split (Optional[Union[str, SplitType]]): Dataset split, defaults to "train".
            **kwargs: Config parameters as keyword arguments.

        Returns:
            Dataset: An instance of the loaded dataset.

        """
        split = split or "train"
        config_filename = config_filename or cls.config_filename
        dataset_config = DatasetConfig.load(hub_path, filename=config_filename, repo_type=RepoType.DATASET)
        dataset_config.path = hub_path
        dataset = build_dataset(dataset_config.name, config=dataset_config, split=split, **kwargs)
        return dataset
