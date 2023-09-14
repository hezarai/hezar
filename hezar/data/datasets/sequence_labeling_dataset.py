from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset

from ...configs import DatasetConfig
from ...constants import TaskType
from ...preprocessors import Tokenizer
from ...registry import register_dataset
from ...utils import Logger
from ..data_collators import SequenceLabelingDataCollator
from .dataset import Dataset


logger = Logger(__name__)


@dataclass
class SequenceLabelingDatasetConfig(DatasetConfig):
    name = "sequence_labeling"
    task: str = TaskType.SEQUENCE_LABELING
    path: str = None
    normalizers: List[Tuple[str, Dict]] = None
    tokenizer_path: str = None
    tags_field: str = None
    tokens_field: str = None
    max_length: int = None
    ignore_index: int = -100
    label_all_tokens: bool = True
    is_iob_schema: bool = False  # Usually set to True for NER & Chunker and set to False for POS


@register_dataset("sequence_labeling", config_class=SequenceLabelingDatasetConfig)
class SequenceLabelingDataset(Dataset):
    """
    A sequence labeling dataset class.
    As of now this class is intended for datasets existing on the Hub!

    Args:
        config: Dataset config object
        split: Which split to use
        **kwargs: Extra config parameters to assign to the original config
    """

    def __init__(self, config: SequenceLabelingDatasetConfig, split=None, **kwargs):
        super().__init__(config, **kwargs)
        self.dataset = self._load(split)
        self._extract_labels()
        self.tokenizer = self._build_tokenizer()
        if self.tokenizer:
            self.data_collator = SequenceLabelingDataCollator(self.tokenizer)

    def _load(self, split):
        """
        Load the dataset

        Args:
            split: Dataset split

        Returns:
            The whole dataset
        """
        # TODO: In case we want to make this class work on other types like csv, json, etc. we have to do it here.
        dataset = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir)
        return dataset

    def _build_tokenizer(self):
        if self.config.tokenizer_path:
            tokenizer = Tokenizer.load(self.config.tokenizer_path)
        else:
            logger.warning("This dataset requires a tokenizer to work. Provide it in config as `tokenizer_path` "
                           "or set it manually as `dataset.tokenizer = your_tokenizer` after building the dataset.")
            tokenizer = None
        return tokenizer

    def _extract_labels(self):
        """
        Extract label names, ids and build dictionaries
        """
        tags_list = self.dataset.features[self.config.tags_field].feature.names
        self.id2label = self.config.id2label = {k: str(v) for k, v in dict(list(enumerate(tags_list))).items()}
        self.label2id = self.config.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(tags_list)

    def __len__(self):
        return len(self.dataset)

    def _tokenize_and_align(self, tokens, labels):
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_word_ids=True,
            padding=True,
            truncation=True,
        )
        word_ids = tokenized_inputs["word_ids"]

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100, so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(self.config.ignore_index)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(labels[word_idx] if self.config.label_all_tokens else self.config.ignore_index)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    def __getitem__(self, index):
        """
        Tokenize inputs and return a dict containing ids, masks, labels, etc.

        Args:
            index: Sample index

        Returns:
            A dict of tokenized text data and labels and some extra stuff
        """
        tokens, tags = self.dataset[index].values()
        inputs = self._tokenize_and_align(tokens, tags)
        return inputs
