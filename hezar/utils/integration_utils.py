import importlib.util
from importlib.metadata import version
from typing import List, Union

from ..constants import Backends


__all__ = [
    "is_backend_available",
    "verify_dependencies",
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


def verify_dependencies(obj, backends: List[Union[Backends, str]] = None):
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
        raise ModuleNotFoundError(
            f"`{obj.__class__.__name__}` requires "
            f"{f'`{unavailable[0]}`' if len(unavailable) == 1 else unavailable} "
            f"which {'is' if len(unavailable) == 1 else 'are'} not installed!"
        )


def get_lib_version(lib: str):
    return version(lib)
