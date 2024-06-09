from dataclasses import dataclass

import torch
from datasets import load_dataset

from ...builders import build_preprocessor
from ...configs import DatasetConfig
from ...constants import Backends, TaskType
from ...preprocessors import ImageProcessorConfig, Tokenizer
from ...registry import register_dataset
from ...utils import Logger, shift_tokens_right
from ..data_collators import ImageCaptioningDataCollator
from .dataset import Dataset


logger = Logger(__name__)

_required_backends = [Backends.SCIKIT]


@dataclass
class ImageCaptioningDatasetConfig(DatasetConfig):
    """
    Configuration class for image captioning datasets.

    Args:
        path (str): Path to the dataset.
        tokenizer_path (str): Path to the tokenizer file.
        text_column (str): Column name for text in the dataset.
        images_paths_column (str): Column name for image paths in the dataset.
        max_length (int): Maximum length of text.
        image_processor_config (ImageProcessorConfig): Configuration for image processing.

    """
    name = "image_captioning"
    task: TaskType = TaskType.IMAGE2TEXT
    path: str = None
    tokenizer_path: str = None
    text_column: str = "label"
    images_paths_column = "image_path"
    max_length: int = None
    image_processor_config: ImageProcessorConfig = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.image_processor_config, dict):
            self.image_processor_config = ImageProcessorConfig(**self.image_processor_config)


@register_dataset("image_captioning", config_class=ImageCaptioningDatasetConfig)
class ImageCaptioningDataset(Dataset):
    required_backends = _required_backends

    def __init__(self, config: ImageCaptioningDatasetConfig, split=None, **kwargs):
        super().__init__(config=config, split=split, **kwargs)
        self.data = self._load(split)
        self.image_processor = build_preprocessor("image_processor", config=self.config.image_processor_config)
        self.tokenizer = Tokenizer.load(self.config.tokenizer_path)
        self.data_collator = ImageCaptioningDataCollator(
            self.tokenizer,
            padding_type="max_length" if self.config.max_length is not None else "longest",
            max_length=self.config.max_length
        )

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
        return data

    def __getitem__(self, index):
        """
        Get a specific item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            dict: The input data.
        """
        path, text = self.data[index].values()
        pixel_values = self.image_processor(path, return_tensors="pt")["pixel_values"]
        tokenized_inputs = self.tokenizer(text, padding="max_length", max_length=self.config.max_length)
        labels = torch.tensor([tokenized_inputs["token_ids"]])
        attention_mask = torch.tensor([tokenized_inputs["attention_mask"]])
        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": attention_mask,
        }
        return inputs
