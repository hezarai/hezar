from collections import defaultdict

import numpy as np
import torch

from ..preprocessors import AudioFeatureExtractor, Tokenizer
from ..utils import Logger, convert_batch_dict_dtype


__all__ = [
    "TextPaddingDataCollator",
    "TextGenerationDataCollator",
    "ImageCaptioningDataCollator",
    "SpeechRecognitionDataCollator",
    "SequenceLabelingDataCollator",
    "CharLevelOCRDataCollator",
]

logger = Logger(__name__)


def _convert_to_batch_dict(dicts_list: list[dict]):
    """
    Convert a list of dicts to a dict of batched values.

    Args:
        dicts_list: A list of dictionaries containing the same set of keys

    Returns:
        A dictionary of the batches
    """
    batch_dict = defaultdict(list)
    for item in dicts_list:
        for key, value in item.items():
            batch_dict[key].append(value)
    batch_dict = dict(batch_dict)
    return batch_dict


class TextPaddingDataCollator:
    """
    A data collator that pads a batch of tokenized inputs.

    Args:
        tokenizer (Tokenizer): A Hezar tokenizer instance.
        padding (str): Specifies padding strategy, either `longest` or `max_length`.
        padding_side (str): Specifies from which side of each tensor to add paddings, either `left` or `right`
        max_length (int): If `padding` is set to `max_length` this must be specified. Forces all tensors to have
            this value as length.
        return_tensors (str): Specifies the dtype of the returning tensors in the batch. (`numpy`, `list`, `torch`)
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: str = "longest",
        padding_side: str = "right",
        max_length: int = None,
        return_tensors: str = "torch",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.padding_side = padding_side
        self.max_length = max_length
        self.return_tensors = return_tensors

        self.field_to_pad_id_mapping = {
            "token_ids": self.tokenizer.pad_token_id,
            "token_type_ids": self.tokenizer.config.pad_token_type_id,
            "tokens": "",
            "special_tokens_mask": 1,
            "attention_mask": 0,
        }

    def __call__(self, input_batch):
        """
        Add padding to every item in the batch

        Args:
            input_batch: A batch dictionary

        Returns:
            Dict: The same batch dictionary but padded
        """
        input_batch = [convert_batch_dict_dtype(x, dtype="list") for x in input_batch]
        input_batch = _convert_to_batch_dict(input_batch)
        labels = input_batch.pop("labels")
        input_length = self.max_length or max(len(x) for x in input_batch["token_ids"])

        for field, batch in input_batch.items():
            padded_batch = []
            for x in batch:
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy().tolist()
                elif isinstance(x, np.ndarray):
                    x = x.tolist()
                difference = input_length - len(x)
                paddings = [self.field_to_pad_id_mapping[field]] * difference
                padded_x = x + paddings if self.padding_side == "right" else paddings + x
                padded_batch.append(padded_x)
            input_batch[field] = padded_batch

        input_batch["labels"] = labels

        input_batch = convert_batch_dict_dtype(input_batch, dtype=self.return_tensors)

        return input_batch


class TextGenerationDataCollator:
    """
    A data collator for text to text generation

    Args:
        tokenizer (Tokenizer): A Hezar tokenizer instance.
        padding (str): Specifies padding strategy, either `longest` or `max_length`.
        padding_side (str): Specifies from which side of each tensor to add paddings, either `left` or `right`
        max_length (int): If `padding` is set to `max_length` this must be specified. Forces all tensors to have
            this value as length.
        labels_max_length (int): Maximum target length for text generation.

    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: str = "longest",
        padding_side: str = "right",
        max_length: int = None,
        labels_max_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.padding_side = padding_side
        self.max_length = max_length
        self.labels_max_length = labels_max_length

    def __call__(self, input_batch):
        """
        Add padding to every item in the batch

        Args:
            input_batch (List[Dict]): A batch dictionary

        Returns:
            Dict: The same batch dictionary but padded
        """
        input_batch = [convert_batch_dict_dtype(x, dtype="list") for x in input_batch]
        input_batch = _convert_to_batch_dict(input_batch)
        padded_batch = self.tokenizer.pad_encoded_batch(
            input_batch,
            padding=self.padding,
            max_length=self.max_length,
            exclude_keys=["labels"],
            return_tensors="torch",
        )
        padded_batch = self.tokenizer.pad_encoded_batch(
            padded_batch,
            padding=self.padding,
            max_length=self.labels_max_length,
            include_keys=["labels"],
            return_tensors="torch",
        )

        return padded_batch


class ImageCaptioningDataCollator:
    """
    Data collator for image captioning.

    Args:
        tokenizer (Tokenizer): A Hezar tokenizer instance.
        padding (str): Specifies padding strategy, either `longest` or `max_length`.
        padding_side (str): Specifies from which side of each tensor to add paddings, either `left` or `right`
        max_length (int): If `padding` is set to `max_length` this must be specified. Forces all tensors to have
            this value as length.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: str = "longest",
        padding_side: str = "right",
        max_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.padding_side = padding_side
        self.max_length = max_length

    def __call__(self, input_batch):
        input_batch = _convert_to_batch_dict(input_batch)
        input_batch = self.tokenizer.pad_encoded_batch(
            input_batch,
            padding=self.padding,
            max_length=self.max_length,
            exclude_keys=["pixel_values"],
            return_tensors="torch",
        )
        if isinstance(input_batch["pixel_values"], list):
            if isinstance(input_batch["pixel_values"][0], list):
                input_batch["pixel_values"] = torch.tensor(input_batch["pixel_values"])
            elif isinstance(input_batch["pixel_values"][0], torch.Tensor):
                input_batch["pixel_values"] = torch.stack(input_batch["pixel_values"])
            elif isinstance(input_batch["pixel_values"][0], np.ndarray):
                input_batch["pixel_values"] = torch.stack([torch.from_numpy(x) for x in input_batch["pixel_values"]])

        return input_batch


class SpeechRecognitionDataCollator:
    def __init__(
        self,
        feature_extractor: AudioFeatureExtractor,
        tokenizer: Tokenizer,
        inputs_padding: str = "longest",
        inputs_max_length: int = None,
        labels_padding: str = "longest",
        labels_max_length: int = None,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.inputs_padding = inputs_padding
        self.inputs_max_length = inputs_max_length
        self.labels_padding = labels_padding
        self.labels_max_length = labels_max_length

    def __call__(self, input_batch):
        input_batch = _convert_to_batch_dict(input_batch)
        inputs = self.tokenizer.pad_encoded_batch(
            input_batch,
            padding=self.labels_padding,
            max_length=self.labels_max_length,
            exclude_keys=["input_features"],
            return_tensors="torch",
        )

        inputs = self.feature_extractor.pad(
            inputs,
            padding=self.inputs_padding,
            max_length=self.inputs_max_length,
            return_tensors="torch",
        )

        return inputs


class SequenceLabelingDataCollator:
    """
    A data collator for sequence labeling.

    Args:
        tokenizer (Tokenizer): A Hezar tokenizer instance.
        padding (str): Specifies padding strategy, either `longest` or `max_length`.
        padding_side (str): Specifies from which side of each tensor to add paddings, either `left` or `right`.
        label_pad_token_id (int): Token ID for padding labels.
        max_length (int): If `padding` is set to `max_length` this must be specified. Forces all tensors to have
            this value as length.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: str = "longest",
        padding_side: str = "right",
        label_pad_token_id: int = -100,
        max_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.padding_side = padding_side
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length

    def __call__(self, input_batch):
        """
        Add padding to every item in the batch

        Args:
            input_batch (List[Dict]): A batch dictionary

        Returns:
            Dict: The same batch dictionary but padded
        """
        input_batch = _convert_to_batch_dict(input_batch)
        labels = input_batch["labels"]
        self.tokenizer.config.padding_side = self.padding_side
        input_batch = self.tokenizer.pad_encoded_batch(
            input_batch,
            padding=self.padding,  # noqa
            max_length=self.max_length,
            return_tensors="torch",
        )

        if labels is None:
            return input_batch

        input_batch.pop("word_ids", None)
        sequence_length = input_batch["token_ids"].shape[1]
        if self.padding_side == "right":
            input_batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            input_batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        input_batch = {
            k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in input_batch.items()
        }

        return input_batch


class CharLevelOCRDataCollator:
    """
    A data collator for character-level OCR.

    Args:
        pad_token_id (int): Token ID for padding characters.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, input_batch):
        """
        Add padding to character-level OCR data.

        Args:
            input_batch (Dict): Input batch containing pixel values and labels.

        Returns:
            Dict: Padded input batch.
        """
        input_batch = _convert_to_batch_dict(input_batch)

        if not isinstance(input_batch["pixel_values"][0], torch.Tensor):
            input_batch["pixel_values"] = torch.tensor(input_batch["pixel_values"])
        elif isinstance(input_batch["pixel_values"], list) and isinstance(input_batch["pixel_values"][0], torch.Tensor):
            input_batch["pixel_values"] = torch.stack(input_batch["pixel_values"])

        max_length = max(map(len, input_batch["labels"]))
        all_labels = []
        for labels in input_batch["labels"]:
            labels += [self.pad_token_id] * (max_length - len(labels))
            all_labels.append(labels)
        input_batch["labels"] = torch.tensor(all_labels)
        return input_batch
