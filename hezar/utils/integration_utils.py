from __future__ import annotations

import importlib.util
from functools import wraps
from importlib.metadata import version
from typing import List

from ..constants import Backends


__all__ = [
    "is_backend_available",
    "verify_dependencies",
    "check_dependencies",
    "get_lib_version",
]


def is_backend_available(backend: Backends):
    """
    Check if the backend package is installed or not

    Args:
        backend: Package name

    Returns:
        Whether the package is available or not
    """
    return importlib.util.find_spec(backend) is not None


def verify_dependencies(obj, backends: List[Backends | str] = None):
    """
    Check if all the required dependencies are installed or not.

    Args:
        obj: The target object to check. (Usually `self`)
        backends: A list of dependency names of type `str` or `Backends`

    Raises:
        ModuleNotFoundError
    """
    if backends is None:
        return
    unavailable = []
    for backend in backends:
        if not is_backend_available(backend):
            unavailable.append(backend)
    if len(unavailable):
        name = obj.__name__ if obj.__class__.__name__ == "function" else obj.__class__.__name__
        raise ModuleNotFoundError(
            f"`{name}` requires {f'`{unavailable[0]}`' if len(unavailable) == 1 else unavailable} "
            f"which {'is' if len(unavailable) == 1 else 'are'} not installed!"
        )


def check_dependencies(backends: list[Backends | str]):
    """
    A wrapper function to verify if the dependencies of the function are installed.

    Args:
        backends: A list of dependencies.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assuming the first argument of the function is the `self` object
            obj = args[0] if args else None
            if obj is not None:
                verify_dependencies(obj, backends)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_lib_version(lib: str):
    return version(lib)
