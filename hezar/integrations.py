import importlib.util

__all__ = [
    "is_transformers_available",
    "is_datasets_available",
    "is_tokenizers_available",
    "is_soundfile_available",
    "is_librosa_available",
    "is_wandb_available",
    "is_gensim_available",
]


def is_transformers_available():
    return importlib.util.find_spec("transformers") is not None


def is_datasets_available():
    return importlib.util.find_spec("datasets") is not None


def is_tokenizers_available():
    return importlib.util.find_spec("tokenizers") is not None


def is_soundfile_available():
    return importlib.util.find_spec("soundfile") is not None


def is_librosa_available():
    return importlib.util.find_spec("librosa") is not None


def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None


def is_gensim_available():
    return importlib.util.find_spec("gensim") is not None
