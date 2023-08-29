from typing import Any, Dict

from .logging import Logger


__all__ = [
    "convert_batch_dict_dtype",
    "get_non_numeric_keys",
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
                batch_dict[k] = v.numpy().tolist()
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
                logger.warning(f"Could not convert values of `{k}` to type `{dtype}`\n"
                               f"Error: {e}")
    return batch_dict


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
