import importlib.util


__all__ = [
    "is_soundfile_available",
    "is_librosa_available",
]


def is_soundfile_available():
    return importlib.util.find_spec("soundfile") is not None


def is_librosa_available():
    return importlib.util.find_spec("librosa") is not None
