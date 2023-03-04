import numpy as np
import torch

from hezar.utils import get_logger
from hezar.data.utils import convert_batch_dict_dtype
from hezar.preprocessors import Tokenizer

__all__ = [
    "TextPaddingDataCollator",
]

logger = get_logger(__name__)


class TextPaddingDataCollator:
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
                f"You passed `max_length` while also setting `padding_type` to `longest` which are "
                f"incompatible! Instead leave `max_length` as None or set `padding_type` to `max_length`! "
                f"Ignoring `max_length`"
            )
            self.max_length = None

    def __call__(self, encoded_batch):
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
