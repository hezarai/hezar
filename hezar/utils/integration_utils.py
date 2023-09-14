import importlib.util

from ..constants import Backends

__all__ = [
    "is_backend_available",
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
