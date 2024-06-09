from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import torch
from datasets import load_dataset

from ...builders import build_preprocessor
from ...configs import DatasetConfig
from ...constants import Backends, TaskType
from ...preprocessors import ImageProcessorConfig, Tokenizer
from ...registry import register_dataset
from ...utils import Logger, is_text_valid, reverse_string_digits
from ..data_collators import CharLevelOCRDataCollator
from .dataset import Dataset


logger = Logger(__name__)

_required_backends = [Backends.SCIKIT]

fa_characters = [
    "", "آ", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د", "ذ", "ر", "ز", "ژ", "س", "ش",
    "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ه", "ی", " "
]
fa_numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹", "۰"]
fa_special_characters = ["ء", "ؤ", "ئ", "أ", "ّ"]
fa_symbols = ["/", "(", ")", "+", "-", ":", "،", "!", ".", "؛", "=", "%", "؟"]
en_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
all_characters = fa_characters + fa_numbers + fa_special_characters + fa_symbols + en_numbers

ID2LABEL = dict(enumerate(all_characters))


class TextSplitType(str, Enum):
    CHAR_SPLIT = "char_split"  # mostly for char level ocr models
    TOKENIZE = "tokenize"  # mostly for transformer-based ocr models


@dataclass
class OCRDatasetConfig(DatasetConfig):
    """
    Configuration class for OCR datasets.

    Args:
        path (str): Path to the dataset.
        text_split_type (TextSplitType): Type of text splitting (CHAR_SPLIT or TOKENIZE).
        tokenizer_path (str): Path to the tokenizer file.
        id2label (Dict[int, str]): Mapping of label IDs to characters.
        text_column (str): Column name for text in the dataset.
        images_paths_column (str): Column name for image paths in the dataset.
        max_length (int): Maximum length of text.
        invalid_characters (list): List of invalid characters.
        reverse_digits (bool): Whether to reverse the digits in text.
        image_processor_config (ImageProcessorConfig): Configuration for image processing.

    """
    name = "ocr"
    task: TaskType = TaskType.IMAGE2TEXT
    path: str = None
    text_split_type: str | TextSplitType = TextSplitType.TOKENIZE
    tokenizer_path: str = None  # if left to None, text_split_type must be `char_split`
    id2label: Dict[int, str] = field(default_factory=lambda: ID2LABEL)
    text_column: str = "label"
    images_paths_column: str = "image_path"
    max_length: int = None
    invalid_characters: list = None
    reverse_text: bool = None
    reverse_digits: bool = None
    image_processor_config: ImageProcessorConfig | dict = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.image_processor_config, dict):
            self.image_processor_config = ImageProcessorConfig(**self.image_processor_config)


@register_dataset("ocr", config_class=OCRDatasetConfig)
class OCRDataset(Dataset):
    """
    General OCR dataset class.

    OCR dataset supports two types of image to text dataset. One is for tokenizer-based models in which the labels are
    tokens and the other is char-level models in which the labels are separated by character and the converted to ids.
    This behavior is specified by the `text_split_type` in config which can be either `tokenize` or `char_split`.

    """
    required_backends = _required_backends

    def __init__(self, config: OCRDatasetConfig, split=None, **kwargs):
        """
        Initializes a new OCRDataset instance.

        Args:
            config (OCRDatasetConfig): The configuration object for the dataset.
            split: Dataset split, defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(config=config, split=split, **kwargs)
        self.data = self._load(split)
        self.image_processor = build_preprocessor("image_processor", config=self.config.image_processor_config)
        if self.config.text_split_type == TextSplitType.TOKENIZE:
            if self.config.tokenizer_path is not None:
                self.tokenizer = Tokenizer.load(self.config.tokenizer_path)
                self.data_collator = None  # TODO resolve this in the future.
            else:
                raise ValueError("No `tokenizer_path` given although `text_split_type` is set to `tokenize`!")
        else:
            self.tokenizer = None
            self.data_collator = CharLevelOCRDataCollator()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.data)

    def _load(self, split=None):
        """
        Load the dataset and clean up invalid samples.

        Args:
            split: Dataset split, defaults to None.

        Returns:
            Dataset: The cleaned dataset.

        """
        data = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir)
        # Cleanup dataset
        valid_indices = []
        invalid_indices = []
        for i, sample in enumerate(list(iter(data))):
            path, text = sample.values()
            if len(text) <= self.config.max_length and is_text_valid(text, self.config.id2label.values()):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)
        if len(invalid_indices):
            logger.warning(
                f"{len(invalid_indices)} invalid samples found in the dataset! "
                f"Inspect them using the `invalid_data` attribute"
            )
        self.invalid_data = data.select(invalid_indices)
        data = data.select(valid_indices)
        return data

    def _text_to_tensor(self, text):
        """
        Convert text to tensor based on the configured text_split_type.

        Args:
            text (str): The raw text.

        Returns:
            torch.Tensor: The output tensor.

        """
        # Tokenize text inputs if text_split_type is set to `tokenize`
        if self.config.text_split_type == TextSplitType.TOKENIZE:
            token_ids = self.tokenizer(text, padding="max_length", max_length=self.config.max_length)["token_ids"]
            # Make sure to ignore pad tokens by the loss function
            token_ids = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in token_ids]
            labels = torch.tensor(token_ids)
        # If split text is not tokenizer-based
        elif self.config.text_split_type == TextSplitType.CHAR_SPLIT:
            if self.config.reverse_digits:
                text = reverse_string_digits(text)
            label2id = {v: k for k, v in self.config.id2label.items()}
            labels = [label2id[x] for x in text]
            labels = torch.LongTensor(labels)
        else:
            raise ValueError(f"Invalid `text_split_type={self.config.text_split_type}`")

        return labels

    def __getitem__(self, index):
        """
        Get a specific item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            dict: The input data.

        """
        path, text = self.data[index].values()
        pixel_values = self.image_processor(path, return_tensors="pt")["pixel_values"][0]
        labels = self._text_to_tensor(text)
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return inputs
