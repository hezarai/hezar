from typing import Dict, List, Any

__all__ = [
    "convert_batch_dict_dtype",
]


def convert_batch_dict_dtype(batch_dict: Dict[str, Any], dtype: str = "list"):
    import numpy as np
    import torch

    if dtype == "list":
        for k, v in batch_dict.items():
            if isinstance(v, np.ndarray):
                batch_dict[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                batch_dict[k] = v.numpy().tolist()
        return batch_dict
    caster = np.ndarray if dtype in ["np", "numpy"] else torch.tensor
    for k, v in batch_dict.items():
        batch_dict[k] = caster(v)
    return batch_dict
