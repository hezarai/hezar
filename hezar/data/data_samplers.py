from typing import Sized

from torch.utils.data import Sampler


__all__ = [
    "SlicedSampler",
]


class SlicedSampler(Sampler):
    """
    A simple dataset sampler that samples the data from a starting index up to an ending index

    Args:
        data: The dataset source
        start_index: The starting index, will default to 0 if not given
        end_index: The ending index, will default to the length of the data source - 1
    """

    def __init__(self, data: Sized, start_index: int = None, end_index: int = None):
        super().__init__()
        self.data = data
        self.start_index = start_index or 0
        self.end_index = end_index or len(self.data)
        self.num_samples = self.end_index - self.start_index
        self.total_length = len(self.data)

    def __len__(self):
        return self.total_length

    def __iter__(self):
        indices = list(range(self.start_index, self.end_index))
        for index in indices:
            if index >= self.start_index:
                yield index
