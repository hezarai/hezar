from typing import List, Union
import importlib.util
from functools import wraps

from ..constants import Backends

__all__ = [
    "is_backend_available",
    "verify_dependencies",
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


def verify_dependencies(obj, backends: List[Union[Backends, str]] = None):
    if backends is None:
        return
    unavailable = []
    for backend in backends:
        if not is_backend_available(backend):
            unavailable.append(backend)
    if len(unavailable):
        raise ModuleNotFoundError(f"`{obj.__class__.__name__}` requires {unavailable} which are not installed!")

