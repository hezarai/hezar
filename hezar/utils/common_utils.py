import inspect
from re import sub
from time import perf_counter
from typing import Callable, Dict, List, Mapping, Union


__all__ = [
    "exec_timer",
    "snake_case",
    "permute_dict_list",
    "sanitize_params_for_fn",
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
    return "_".join(sub("([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))).split()).lower()


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


def sanitize_params_for_fn(fn: Callable, params: Union[Dict, Mapping], **kwargs):
    """
    Given a dict of parameters or kwargs, you can figure out which ones must be passed to the `fn` based on its
    signature.

    Args:
         fn: The function object
         params: A dict of parameters with values
         kwargs: Keyword arguments that are merged with `params`

    Returns:
        The proper dict of parameters keys and values
    """
    params.update(**kwargs)
    params_signature = dict(inspect.signature(fn).parameters)
    fn_parameters = {p for p, v in params_signature.items() if v.kind not in (v.VAR_KEYWORD, v.VAR_POSITIONAL)}
    fn_params_names = set(fn_parameters)
    input_params = {p: params[p] for p in fn_params_names if p in params}
    return input_params
