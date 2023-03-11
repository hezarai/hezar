import os


HEZAR_HUB_ID = "hezar-ai"
HEZAR_CACHE_DIR = os.getenv("HEZAR_CACHE_DIR", f'{os.path.expanduser("~")}/.hezar')
HEZAR_TMP_DIR = os.getenv("HEZAR_TMP_DIR", f'{os.path.expanduser("~")}/.cache/hezar')
HEZAR_SNAPSHOTS_DIR = os.getenv("HEZAR_SNAPSHOTS_DIR", f"{HEZAR_CACHE_DIR}/snapshots")
HEZAR_MODELS_CACHE_DIR = os.getenv("HEZAR_MODELS_CACHE_DIR", f"{HEZAR_CACHE_DIR}/models")
HEZAR_DATASETS_CACHE_DIR = os.getenv("HEZAR_DATASETS_CACHE_DIR", f"{HEZAR_CACHE_DIR}/datasets")
REPO_TYPE_TO_DIR_MAPPING = {"model": HEZAR_MODELS_CACHE_DIR, "dataset": HEZAR_DATASETS_CACHE_DIR}

DEFAULT_MODEL_FILE = "model.pt"
DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_TRAINER_SUBFOLDER = "train"
DEFAULT_TRAINER_CONFIG_FILE = "train_config.yaml"
DEFAULT_PREPROCESSOR_SUBFOLDER = "preprocessor"
DEFAULT_TOKENIZER_FILE = "tokenizer.json"
DEFAULT_TOKENIZER_CONFIG_FILE = "tokenizer_config.yaml"
