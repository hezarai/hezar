from dataclasses import dataclass

import torch

from ...configs import DatasetConfig
from ...constants import Backends, TaskType
from ...registry import register_dataset
from ...utils import Logger, is_backend_available
from ..data_collators import TextPaddingDataCollator
from .dataset import Dataset


if is_backend_available(Backends.DATASETS):
    from datasets import load_dataset

logger = Logger(__name__)


@dataclass
class TextClassificationDatasetConfig(DatasetConfig):
    """
    Configuration class for text classification datasets.

    Args:
        path (str): Path to the dataset.
        label_field (str): Field name for labels in the dataset.
        text_field (str): Field name for text in the dataset.
        max_length (int): Maximum length of text.
    """

    name = "text_classification"
    task: TaskType = TaskType.TEXT_CLASSIFICATION
    path: str = None
    label_field: str = None
    text_field: str = None
    max_length: int = None


@register_dataset("text_classification", config_class=TextClassificationDatasetConfig)
class TextClassificationDataset(Dataset):
    """
    A text classification dataset class.
    As of now this class is intended for datasets existing on the Hub!

    Args:
        config (TextClassificationDatasetConfig): Dataset config object.
        split: Which split to use.
        preprocessor: Dataset's preprocessor
        **kwargs: Extra config parameters to assign to the original config.
    """

    def __init__(self, config: TextClassificationDatasetConfig, split=None, preprocessor=None, **kwargs):
        super().__init__(config, split=split, preprocessor=preprocessor, **kwargs)
        self._extract_labels()
        self.tokenizer = self.preprocessor.tokenizer
        self.data_collator = TextPaddingDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        ) if self.tokenizer else None

    def _load(self, split):
        """
        Load the dataset.

        Args:
            split: Dataset split.

        Returns:
            The whole dataset.

        """
        dataset = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir, **self.config.hf_load_kwargs)
        return dataset

    def _extract_labels(self):
        """
        Extract label names, ids and build dictionaries.
        """
        labels_list = self.data.features[self.config.label_field].names
        self.id2label = self.config.id2label = {k: str(v) for k, v in dict(enumerate(labels_list)).items()}
        self.label2id = self.config.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(labels_list)

    def __getitem__(self, index):
        """
        Tokenize inputs and return a dict containing ids, masks, labels, etc.

        Args:
            index: Sample index.

        Returns:
            dict: The input data.

        """
        text = self.data[index][self.config.text_field]
        label = self.data[index][self.config.label_field]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation_strategy="longest_first",
            padding="longest",
            return_attention_mask=True,
        )
        label_idx = torch.tensor([label], dtype=torch.long)  # noqa
        inputs["labels"] = label_idx

        return inputs
