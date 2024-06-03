from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from omegaconf import DictConfig

from ..constants import PaddingType
from .logging import Logger


if TYPE_CHECKING:
    import torch

__all__ = [
    "convert_batch_dict_dtype",
    "resolve_inputs_length_for_padding",
    "pad_batch_items",
    "shift_tokens_right",
    "torch2numpy",
    "get_non_numeric_keys",
    "flatten_dict",
    "set_seed",
    "dataloader_worker_init_fn",
]

logger = Logger(__name__)


# TODO: This code might be able to be written in a cleaner way, but be careful, any change might break a lot of things!
def convert_batch_dict_dtype(batch_dict: Dict[str, Any], dtype: str = None, skip_keys: list = None):
    """
    Convert data dtypes of the values in a batch dict

    Args:
        batch_dict: The batched dictionary. Each key in the dict has a batch of data as its value.
        dtype: Target data type to convert to
        skip_keys: A list of key names to skip conversion

    Returns:
        The same dict with cast values
    """
    import numpy as np
    import torch

    dtype = dtype or "list"
    skip_keys = skip_keys or []

    if dtype == "list":
        for k, v in batch_dict.items():
            if isinstance(v, np.ndarray):
                batch_dict[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                batch_dict[k] = v.cpu().numpy().tolist()
        return batch_dict

    if dtype in ["np", "numpy"]:
        caster = np.asarray
        cast_type = np.ndarray
    elif dtype in ["pt", "torch", "pytorch"]:
        caster = torch.tensor
        cast_type = torch.Tensor
    else:
        raise ValueError(f"Invalid `dtype`: {dtype}")

    for k, v in batch_dict.items():
        if k not in skip_keys:
            try:
                if not isinstance(v, cast_type):
                    batch_dict[k] = caster(v)
            except Exception as e:  # noqa
                logger.warning(f"Could not convert values of `{k}` to type `{dtype}`\n" f"Error: {e}")
    return batch_dict


def resolve_inputs_length_for_padding(
    inputs: List[List[Any]],
    padding_type: str | PaddingType = None,
    max_length: Optional[bool | int] = None,
    truncation: Optional[bool] = True,
):
    """
    Resolve final inputs length based on padding_strategy and max_length values
    """
    inputs_max_length = max([len(x) for x in inputs])

    # Resolve padding and max_length values first
    if padding_type is None:
        if max_length is not None:
            padding_type = "max_length"
        else:
            padding_type = "longest"

    # Now lets resolve any conflicts
    if padding_type == "longest":
        if max_length is not None and max_length < inputs_max_length:
            if not truncation:
                logger.warning(
                    f"Setting padding_type to `longest` and a max_length={max_length} which is below the "
                    f"inputs longest value {inputs_max_length} requires truncation to be True, "
                    f"got truncation={truncation}. Falling back to truncating inputs to max_length={max_length}."
                )
                inputs_length = max_length
            else:
                inputs_length = max_length
        else:
            inputs_length = inputs_max_length

    elif padding_type == "max_length":
        if max_length is None:
            logger.warning(
                "Setting padding='max_length' but no max_length value is provided! Falling back to padding='longest'"
            )
            inputs_length = inputs_max_length
        else:
            if max_length < inputs_max_length and not truncation:
                logger.warning(
                    f"Cannot set max_length to {max_length} "
                    f"while max input length is {inputs_max_length} and `truncation` is `False`"
                    f"Either set `truncation=True` or increase `max_length`"
                )
                inputs_length = inputs_max_length
            else:
                inputs_length = max_length
    else:
        raise ValueError(f"Invalid padding value `{padding_type}`, expected either `max_length` or `longest`")

    return inputs_length


def pad_batch_items(
    inputs: List[List[int | float]],
    padding_type: str | PaddingType = None,
    padding_side: Literal["right", "left"] = "right",
    pad_id: int = 0,
    max_length: Optional[bool | int] = None,
    truncation: Optional[bool] = True,
):
    """
    Given a nested container of unequal sized iterables e.g, batch of token ids, pad them based on padding strategy
    Args:
        inputs: A nested iterable of unequal sized iterables (e.g, list of lists)
        padding_type: Padding strategy, either max_length or longest
        padding_side: Where to add padding ids, `left` or `right`, defaults to `right`
        pad_id: Pad token id, defaults to `0`
        max_length: Max input length after padding, only applicable when padding_strategy == "max_length"
        truncation: Whether to truncate if an input in the batch is longer than max_length

    Returns:
        A list of equal sized lists
    """

    inputs_length = resolve_inputs_length_for_padding(
        inputs,
        padding_type=padding_type,
        max_length=max_length,
        truncation=truncation,
    )

    padded_inputs = []
    for ids in inputs:
        difference = inputs_length - len(ids)
        if difference > 0:
            paddings = [pad_id] * difference
            padded_ids = ids + paddings if padding_side == "right" else paddings + ids
            padded_inputs.append(padded_ids)
        else:
            padded_inputs.append(ids[:inputs_length])

    return padded_inputs


def shift_tokens_right(input_ids: "torch.Tensor", pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def torch2numpy(*args):
    """
    Cast tensors to numpy

    Args:
        *args: Any number of torch.Tensor objects

    Returns:
        The same inputs cast to numpy
    """
    return [arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]


def get_non_numeric_keys(d: Dict, batched=True):
    """
    Get keys that have string values in a dictionary

    Args:
        d: The dict
        batched: Are the input dict values batched or not

    Returns:
        A list of string-valued keys
    """
    keys = []
    for k, v in d.items():
        if len(v) and isinstance(v[0], list):
            if batched and not isinstance(v[0][0], (int, float, complex)) and not isinstance(v[0][0], bool):
                keys.append(k)
            elif isinstance(v[0], str):
                keys.append(k)
    return keys


def flatten_dict(dict_config: Dict | DictConfig) -> DictConfig:
    """
    Flatten a nested Dict/DictConfig object

    Args:
        dict_config: A Dict/DictConfig object

    Returns:
        The flattened version of the dict-like object
    """

    config = DictConfig({})
    for k, v in dict_config.items():
        if isinstance(v, (Dict, DictConfig)):
            config.update(flatten_dict(v))
        else:
            config[k] = v

    return config


def set_seed(seed):
    """
    Set a global seed for all backends to handle reproducibility and determinism.
    """
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def dataloader_worker_init_fn(seed):
    """
    A dataloader worker init function that handles reproducibility by hard-setting the seed for all workers.
    """

    def worker_init_fn(worker_id, seed):
        set_seed(seed + worker_id)

    return partial(worker_init_fn, seed=seed)
