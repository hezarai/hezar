import importlib


__all__ = [
    "is_soundfile_available",
]


def is_soundfile_available():
    try:
        importlib.import_module("soundfile")
    except ImportError:
        raise ImportError("soundfile is not installed!")
