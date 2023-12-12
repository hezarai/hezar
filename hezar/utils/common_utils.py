from __future__ import annotations

import inspect
import re
from time import perf_counter
from typing import Callable, Dict, List, Mapping

from ..constants import Color


__all__ = [
    "exec_timer",
    "snake_case",
    "reverse_string_digits",
    "is_text_valid",
    "is_url",
    "colorize_text",
    "permute_dict_list",
    "sanitize_function_parameters",
]


class exec_timer:
    """
    A context manager that captures the execution time of all the operations inside it

    Examples:
        >>> with exec_timer() as timer:
        >>>     # operations here
        >>> print(timer.time)
    """

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time


def snake_case(s):
    return "_".join(re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s.replace("-", " "))).split()).lower()


def reverse_string_digits(text):
    """
    Reverse all digit segments in a given text
    """
    # Capture one or more digits followed by any number of non-digits followed by another digit
    pattern = r"(\d+(?:\D\d+)*)"

    def reverse_match(match):
        return match.group(1)[::-1]  # Reverse the matched digits and special characters

    return re.sub(pattern, reverse_match, text)


def is_text_valid(text, valid_characters):
    """
    Given a list of valid characters, check if only those are included in the text
    """
    pattern = re.compile(f'^[{re.escape("".join(valid_characters))}]+$')
    return bool(pattern.match(text))


def is_url(text):
    url_pattern = re.compile(
        r'^(https?|ftp)://'  # Protocol (http, https, ftp)
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # Domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4
        r'\[?[A-F0-9]*:[A-F0-9:]+]?)'  # IPv6
        r'(?::\d+)?'  # Port
        r'(?:/?\S*)?$', re.IGNORECASE
    )
    return bool(re.match(url_pattern, text))


def colorize_text(text: str, color: str | Color):
    """
    Add colorization codes to the text. The output is the text with surrounding color codes and the colors are applied
    on the console/terminal output like when using `print()`
    """
    color_mapping = {
        "header": Color.HEADER,
        "normal": Color.NORMAL,
        "bold": Color.BOLD,
        "underline": Color.UNDERLINE,
        "italic": Color.ITALIC,
        "blue": Color.BLUE,
        "cyan": Color.CYAN,
        "green": Color.GREEN,
        "yellow": Color.YELLOW,
        "red": Color.RED,
        "grey": Color.GREY,
    }
    if isinstance(color, str) and not hasattr(color, "value"):
        color = color_mapping.get(color.lower(), Color.NORMAL)
    return color + text + Color.NORMAL


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


def sanitize_function_parameters(func: Callable, params: Dict | Mapping, **kwargs):
    """
    Given a dict of parameters or kwargs, you can figure out which ones must be passed to the `func` based on its
    signature.

    Args:
         func: The function object
         params: A dict of parameters with values
         kwargs: Keyword arguments that are merged with `params`

    Returns:
        The proper dict of parameters keys and values
    """
    params.update(**kwargs)
    params_signature = dict(inspect.signature(func).parameters)
    fn_parameters = {p for p, v in params_signature.items() if v.kind not in (v.VAR_KEYWORD, v.VAR_POSITIONAL)}
    fn_params_names = set(fn_parameters)
    input_params = {p: params[p] for p in fn_params_names if p in params}
    return input_params

