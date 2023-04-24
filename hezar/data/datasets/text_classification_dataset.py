from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset

from ...configs import DatasetConfig
from ...preprocessors import Sequential, Tokenizer
from ...registry import register_dataset
from ..data_collators import TextPaddingDataCollator
from .dataset import Dataset


@dataclass
class TextClassificationDatasetConfig(DatasetConfig):
    name: str = "text_classification"
    task: str = "text_classification"
    path: str = None
    normalizers: List[Tuple[str, Dict]] = None
    tokenizer_path: str = None
    label_field: str = None
    text_field: str = None
    max_length: int = None


@register_dataset("text_classification", config_class=TextClassificationDatasetConfig)
class TextClassificationDataset(Dataset):
    """
    A text classification dataset class.
    As of now this class is intended for datasets existing on the Hub!

    Args:
        config: Dataset config obj
        split: Which split to use
        **kwargs: Extra config parameters to assign to the original config
    """

    def __init__(self, config: TextClassificationDatasetConfig, split=None, **kwargs):
        super().__init__(config, **kwargs)
        self.dataset = self._load(split)
        self._extract_labels()
        self.tokenizer = Tokenizer.load(self.config.tokenizer_path)
        self.normalizer = Sequential(self.config.normalizers)
        self.data_collator = TextPaddingDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

    def _load(self, split):
        """
        Load the dataset

        Args:
            split: Dataset split

        Returns:
            The whole dataset
        """
        # TODO: In case we want to make this class work on other types like csv, json, etc. we have to do it here.
        dataset = load_dataset(self.config.path, split=split)
        return dataset

    def _extract_labels(self):
        """
        Extract label names, ids and build dictionaries
        """
        labels_list = self.dataset.features[self.config.label_field].names
        self.id2label = self.config.id2label = {k: str(v) for k, v in dict(list(enumerate(labels_list))).items()}
        self.label2id = self.config.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(labels_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Tokenize inputs and return a dict containing ids, masks, labels, etc.

        Args:
            index: Sample index

        Returns:
            A dict of tokenized text data and labels and some extra stuff
        """
        text = self.dataset[index][self.config.text_field]
        label = self.dataset[index][self.config.label_field]
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
