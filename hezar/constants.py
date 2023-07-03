"""
Home to all constant variables in Hezar
"""

import os
from enum import Enum


HEZAR_HUB_ID = "hezarai"
HEZAR_CACHE_DIR = os.getenv("HEZAR_CACHE_DIR", f'{os.path.expanduser("~")}/.hezar')

DEFAULT_MODEL_FILE = "model.pt"
DEFAULT_MODEL_CONFIG_FILE = "model_config.yaml"
DEFAULT_TRAINER_SUBFOLDER = "train"
DEFAULT_TRAINER_CONFIG_FILE = "train_config.yaml"
DEFAULT_PREPROCESSOR_SUBFOLDER = "preprocessor"
DEFAULT_TOKENIZER_FILE = "tokenizer.json"
DEFAULT_TOKENIZER_CONFIG_FILE = "tokenizer_config.yaml"
DEFAULT_DATASET_CONFIG_FILE = "dataset_config.yaml"
DEFAULT_EMBEDDING_FILE = "embedding.bin"
DEFAULT_EMBEDDING_CONFIG_FILE = "embedding_config.yaml"
DEFAULT_EMBEDDING_SUBFOLDER = "embedding"

TQDM_BAR_FORMAT = "{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}"


class TaskType(Enum):
    AUDIO_CLASSIFICATION = "audio_classification"
    IMAGE2TEXT = "image2text"
    LANGUAGE_MODELING = "language_modeling"
    SEQUENCE_LABELING = "sequence_labeling"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_DETECTION = "text_detection"
    SEQ2SEQ = "seq2seq"
