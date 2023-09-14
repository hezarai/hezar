import numpy as np
import torch

from ..preprocessors import Tokenizer
from ..utils import Logger, convert_batch_dict_dtype


__all__ = [
    "TextPaddingDataCollator",
    "SequenceLabelingDataCollator",
]

logger = Logger(__name__)


class TextPaddingDataCollator:
    """
    A data collator that pads a batch of tokenized inputs.

    Args:
        tokenizer: A Hezar tokenizer instance. (only its config is going to be used)
        padding_type: Specifies padding strategy. Defaults to `longest`, but can also be `max_length` (in this case
         `max_length` cannot be None)
        padding_side: Specifies from which side of each tensor to add paddings. Defaults to `right`, but can also be
         `left`.
        max_length: If `padding_type` is set to `max_length` this parameter must be specified. Forces all tensors to
         have this value as length.
        return_tensors: Specifies the dtype of the returning tensors in the batch. Defaults to `pt(torch.Tensor)`, but
         can also be `np` or `list`.

    """

    def __init__(
            self,
            tokenizer: Tokenizer,
            padding_type: str = "longest",
            padding_side: str = "right",
            max_length: int = None,
            return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding_type = padding_type
        self.padding_side = padding_side
        self.max_length = max_length
        self.return_tensors = return_tensors

        self.field_to_pad_id_mapping = {
            "token_ids": self.tokenizer.config.pad_token_id,
            "token_type_ids": self.tokenizer.config.pad_token_type_id,
            "tokens": "",
            "special_tokens_mask": 1,
            "attention_mask": 0,
        }

        if padding_type == "longest" and max_length is not None:
            logger.warning(
                "You passed `max_length` while also setting `padding_type` to `longest` which are "
                "incompatible! Instead leave `max_length` as None or set `padding_type` to `max_length`! "
                "Ignoring `max_length`"
            )
            self.max_length = None

    def __call__(self, encoded_batch):
        """
        Add padding to every item in the batch

        Args:
            encoded_batch: A batch dictionary

        Returns:
            The same batch dictionary but padded
        """
        encoded_batch = [convert_batch_dict_dtype(x, dtype="list") for x in encoded_batch]
        permuted_batch = {}
        for key in encoded_batch[0].keys():
            stack = [e for item in encoded_batch for e in item[key]]
            permuted_batch[key] = stack

        encoded_batch = permuted_batch.copy()
        if "label" in encoded_batch:
            encoded_batch["labels"] = encoded_batch["label"]
            del encoded_batch["label"]

        labels = encoded_batch.pop("labels")
        input_length = self.max_length or max(len(x) for x in encoded_batch["token_ids"])

        for field, batch in encoded_batch.items():
            padded_batch = []
            for x in batch:
                if isinstance(x, torch.Tensor):
                    x = x.numpy().tolist()
                elif isinstance(x, np.ndarray):
                    x = x.tolist()
                difference = input_length - len(x)
                paddings = [self.field_to_pad_id_mapping[field]] * difference
                padded_x = x + paddings if self.padding_side == "right" else paddings + x
                padded_batch.append(padded_x)
            encoded_batch[field] = padded_batch

        encoded_batch["labels"] = labels

        encoded_batch = convert_batch_dict_dtype(encoded_batch, dtype=self.return_tensors)

        return encoded_batch


class SequenceLabelingDataCollator:
    """
    A data collator for sequence labeling.

    Args:
        tokenizer: A Hezar tokenizer instance. (only its config is going to be used)
        padding_type: Specifies padding strategy. Defaults to `longest`, but can also be `max_length` (in this case
         `max_length` cannot be None)
        padding_side: Specifies from which side of each tensor to add paddings. Defaults to `right`, but can also be
         `left`.
        max_length: If `padding_type` is set to `max_length` this parameter must be specified. Forces all tensors to
         have this value as length.
        return_tensors: Specifies the dtype of the returning tensors in the batch. Defaults to `pt(torch.Tensor)`, but
         can also be `np` or `list`.

    """

    def __init__(
            self,
            tokenizer: Tokenizer,
            padding_type: str = "longest",
            padding_side: str = "right",
            label_pad_token_id: int = -100,
            max_length: int = None,
            return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding_type = padding_type
        self.padding_side = padding_side
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, encoded_batch):
        label_name = "label" if "label" in encoded_batch[0].keys() else "labels"
        labels = [feature[label_name] for feature in encoded_batch] if label_name in encoded_batch[0].keys() else None
        self.tokenizer.config.padding_direction = self.padding_side
        batch = self.tokenizer.pad_encoded_batch(
            encoded_batch,
            padding=self.padding_type,  # noqa
            max_length=self.max_length,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        batch.pop("word_ids")
        sequence_length = torch.tensor(batch["token_ids"]).shape[1]
        if self.padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
