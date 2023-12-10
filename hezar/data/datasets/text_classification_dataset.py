from dataclasses import dataclass

import torch
from datasets import load_dataset

from ...configs import DatasetConfig
from ...constants import TaskType
from ...preprocessors import Tokenizer
from ...registry import register_dataset
from ...utils import Logger
from ..data_collators import TextPaddingDataCollator
from .dataset import Dataset


logger = Logger(__name__)


@dataclass
class TextClassificationDatasetConfig(DatasetConfig):
    """
    Configuration class for text classification datasets.

    Args:
        path (str): Path to the dataset.
        tokenizer_path (str): Path to the tokenizer file.
        label_field (str): Field name for labels in the dataset.
        text_field (str): Field name for text in the dataset.
        max_length (int): Maximum length of text.
    """

    name = "text_classification"
    task: TaskType = TaskType.TEXT_CLASSIFICATION
    path: str = None
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
        config (TextClassificationDatasetConfig): Dataset config object.
        split: Which split to use.
        **kwargs: Extra config parameters to assign to the original config.
    """

    def __init__(self, config: TextClassificationDatasetConfig, split=None, **kwargs):
        """
        Initializes a new TextClassificationDataset instance.

        Args:
            config (TextClassificationDatasetConfig): The configuration object for the dataset.
            split: Dataset split, defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(config, split=split, **kwargs)
        self.dataset = self._load(split)
        self._extract_labels()
        self.tokenizer = self._build_tokenizer()
        self.data_collator = TextPaddingDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

    def _load(self, split):
        """
        Load the dataset.

        Args:
            split: Dataset split.

        Returns:
            The whole dataset.

        """
        # TODO: In case we want to make this class work on other types like csv, json, etc. we have to do it here.
        dataset = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir)
        return dataset

    def _build_tokenizer(self):
        """
        Build the tokenizer.

        Returns:
            Tokenizer: The tokenizer.

        """
        if self.config.tokenizer_path:
            tokenizer = Tokenizer.load(self.config.tokenizer_path)
        else:
            logger.warning(
                "This dataset requires a tokenizer to work. Provide it in config as `tokenizer_path` "
                "or set it manually as `dataset.tokenizer = your_tokenizer` after building the dataset."
            )
            tokenizer = None
        return tokenizer

    def _extract_labels(self):
        """
        Extract label names, ids and build dictionaries.
        """
        labels_list = self.dataset.features[self.config.label_field].names
        self.id2label = self.config.id2label = {k: str(v) for k, v in dict(enumerate(labels_list)).items()}
        self.label2id = self.config.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(labels_list)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Tokenize inputs and return a dict containing ids, masks, labels, etc.

        Args:
            index: Sample index.

        Returns:
            dict: The input data.

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
