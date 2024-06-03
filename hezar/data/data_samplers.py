import random
from typing import Sized

from torch.utils.data import Sampler

from ..utils import set_seed


__all__ = [
    "RangedSampler",
]


class RangedSampler(Sampler):
    """
    A simple dataset sampler that samples the data from a starting index up to an ending index and also supports
    shuffling, drop_last, etc.

    Args:
        data: The dataset source
        batch_size: Batch size
        start_index: The starting index, will default to 0 if not given
        end_index: The ending index, will default to the length of the data source
        drop_last: Whether to drop the indices of the last batch
        shuffle: Whether to shuffle indices
        seed: Seed value for shuffling. Required if shuffle is True
    """

    def __init__(
        self,
        data: Sized,
        batch_size: int,
        start_index: int = None,
        end_index: int = None,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = None
    ):
        super().__init__()
        self.data = data
        self.start_index = start_index or 0
        self.end_index = end_index or len(self.data)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.end_index = (self.end_index // batch_size) * batch_size if self.drop_last else self.end_index
        self.num_samples = self.end_index - self.start_index
        self.total_length = len(self.data)

        self.indices = self._create_indices()

    def _create_indices(self):
        indices = list(range(self.total_length))
        if self.shuffle:
            if self.seed is None:
                raise ValueError("Parameter `seed` is required to enable shuffling in the data sampler!")
            set_seed(self.seed)
            random.shuffle(indices)
        indices = indices[self.start_index:self.end_index]
        return indices

    def __len__(self):
        return self.total_length

    def __iter__(self):
        for indice in self.indices:
            yield indice
