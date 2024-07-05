from dataclasses import dataclass

import torch

from ...configs import DatasetConfig
from ...constants import Backends, TaskType
from ...registry import register_dataset
from ...utils import Logger, is_backend_available, shift_tokens_right
from ..data_collators import ImageCaptioningDataCollator
from .dataset import Dataset


if is_backend_available(Backends.DATASETS):
    from datasets import load_dataset

logger = Logger(__name__)

_required_backends = [Backends.DATASETS]


@dataclass
class ImageCaptioningDatasetConfig(DatasetConfig):
    """
    Configuration class for image captioning datasets.

    Args:
        path (str): Path to the dataset.
        text_column (str): Column name for text in the dataset.
        images_paths_column (str): Column name for image paths in the dataset.
        max_length (int): Maximum length of text.

    """
    name = "image_captioning"
    task: TaskType = TaskType.IMAGE2TEXT
    path: str = None
    text_column: str = "label"
    images_paths_column = "image_path"
    max_length: int = None


@register_dataset("image_captioning", config_class=ImageCaptioningDatasetConfig)
class ImageCaptioningDataset(Dataset):
    required_backends = _required_backends

    def __init__(self, config: ImageCaptioningDatasetConfig, split=None, preprocessor=None, **kwargs):
        super().__init__(config=config, split=split, preprocessor=preprocessor, **kwargs)
        self.image_processor = self.preprocessor.image_processor
        self.tokenizer = self.preprocessor.tokenizer
        self.data_collator = ImageCaptioningDataCollator(
            self.tokenizer,
            padding_type="max_length" if self.config.max_length is not None else "longest",
            max_length=self.config.max_length
        )

    def _load(self, split):
        """
        Load the dataset and clean up invalid samples.

        Args:
            split: Dataset split, defaults to None.

        Returns:
            Dataset: The cleaned dataset.

        """
        data = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir, **self.config.hf_load_kwargs)
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
