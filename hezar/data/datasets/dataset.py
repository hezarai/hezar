import os
from typing import Optional, Union

from torch.utils.data import Dataset as TorchDataset

from ...builders import build_dataset
from ...configs import DatasetConfig
from ...constants import DEFAULT_DATASET_CONFIG_FILE, HEZAR_CACHE_DIR, RepoType, SplitType


class Dataset(TorchDataset):
    """
    Base class for all datasets in Hezar.

    Args:
        config:
        **kwargs:
    """
    config_filename = DEFAULT_DATASET_CONFIG_FILE
    cache_dir = HEZAR_CACHE_DIR

    def __init__(self, config: DatasetConfig, **kwargs):
        self.config = config.update(kwargs)
        self.preprocessor = None
        self.data_collator = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
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
        Load the dataset from hub.

        Args:
            hub_path: Path to dataset from hub or locally
            config_filename: Dataset config file name
            split: Dataset split, defaults to "train"
            **kwargs: Config parameters as keyword arguments

        Returns:

        """
        split = split or "train"
        config_filename = config_filename or cls.config_filename
        dataset_config = DatasetConfig.load(hub_path, filename=config_filename, repo_type=RepoType.DATASET)
        dataset_config.path = hub_path
        dataset = build_dataset(dataset_config.name, config=dataset_config, split=split, **kwargs)
        return dataset
