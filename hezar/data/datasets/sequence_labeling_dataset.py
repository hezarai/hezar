from dataclasses import dataclass

from ...configs import DatasetConfig
from ...constants import Backends, TaskType
from ...registry import register_dataset
from ...utils import Logger, is_backend_available
from ..data_collators import SequenceLabelingDataCollator
from .dataset import Dataset


if is_backend_available(Backends.DATASETS):
    from datasets import load_dataset

logger = Logger(__name__)


@dataclass
class SequenceLabelingDatasetConfig(DatasetConfig):
    """
    Configuration class for sequence labeling datasets.

    Args:
        path (str): Path to the dataset.
        tags_field (str): Field name for tags in the dataset.
        tokens_field (str): Field name for tokens in the dataset.
        max_length (int): Maximum length of tokens.
        ignore_index (int): Index to ignore in the loss function.
        label_all_tokens (bool): Whether to label all tokens or just the first token in a word.
        is_iob_schema (bool): Whether the dataset follows the IOB schema.
    """

    name = "sequence_labeling"
    task: TaskType = TaskType.SEQUENCE_LABELING
    path: str = None
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
        config (SequenceLabelingDatasetConfig): Dataset config object.
        split: Which split to use.
        **kwargs: Extra config parameters to assign to the original config.
    """

    def __init__(self, config: SequenceLabelingDatasetConfig, split=None, preprocessor=None, **kwargs):
        super().__init__(config, split=split, preprocessor=preprocessor, **kwargs)
        self._extract_labels()
        self.tokenizer = self.preprocessor.tokenizer
        self.data_collator = SequenceLabelingDataCollator(self.tokenizer, max_length=self.config.max_length)

    def _load(self, split):
        """
        Load the dataset.

        Args:
            split: Dataset split.

        Returns:
            The whole dataset.

        """
        data = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir, **self.config.hf_load_kwargs)
        return data

    def _extract_labels(self):
        """
        Extract label names, ids and build dictionaries.
        """
        tags_list = self.data.features[self.config.tags_field].feature.names
        self.id2label = self.config.id2label = {k: str(v) for k, v in dict(enumerate(tags_list)).items()}
        self.label2id = self.config.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(tags_list)

    def _tokenize_and_align(self, tokens, labels):
        """
        Tokenize and align tokens and labels.

        Args:
            tokens: List of tokens.
            labels: List of labels.

        Returns:
            dict: Tokenized and aligned inputs.

        """
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
            index: Sample index.

        Returns:
            dict: The input data.

        """
        tokens, tags = self.data[index].values()
        inputs = self._tokenize_and_align(tokens, tags)
        return inputs
