from torch.utils.data import Dataset as TorchDataset

from ...configs import DatasetConfig


class Dataset(TorchDataset):
    def __init__(self, config: DatasetConfig, **kwargs):
        self.config = config.update(kwargs)
        self.preprocessor = None

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
