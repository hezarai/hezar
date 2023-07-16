from re import sub
from typing import Dict, List


__all__ = [
    "snake_case",
    "permute_dict_list",
]


def snake_case(s):
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                s.replace('-', ' '))).split()).lower()


def permute_dict_list(dict_list: List[Dict]) -> Dict[str, List]:
    """
    Convert a list of dictionaries to a dictionary of lists

    Args:
        dict_list: Input list of dicts

    Returns:

    """
    if not len(dict_list):
        return {}
    d = {key: [x[key] for x in dict_list] for key in dict_list[0]}
    return d
