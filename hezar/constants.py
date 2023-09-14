"""
Home to all constant variables in Hezar
"""

import os
from enum import Enum


HEZAR_HUB_ID = "hezarai"
HEZAR_CACHE_DIR = os.getenv("HEZAR_CACHE_DIR", f'{os.path.expanduser("~")}/.cache/hezar')

DEFAULT_MODEL_FILE = "model.pt"
DEFAULT_MODEL_CONFIG_FILE = "model_config.yaml"
DEFAULT_TRAINER_SUBFOLDER = "train"
DEFAULT_TRAINER_CONFIG_FILE = "train_config.yaml"
DEFAULT_TRAINER_CSV_LOG_FILE = "training_logs.csv"
DEFAULT_PREPROCESSOR_SUBFOLDER = "preprocessor"
DEFAULT_NORMALIZER_CONFIG_FILE = "normalizer_config.yaml"
DEFAULT_IMAGE_PROCESSOR_CONFIG_FILE = "image_processor_config.yaml"
DEFAULT_FEATURE_EXTRACTOR_CONFIG_FILE = "feature_extractor_config.yaml"
DEFAULT_TOKENIZER_FILE = "tokenizer.json"
DEFAULT_TOKENIZER_CONFIG_FILE = "tokenizer_config.yaml"
DEFAULT_DATASET_CONFIG_FILE = "dataset_config.yaml"
DEFAULT_EMBEDDING_FILE = "embedding.bin"
DEFAULT_EMBEDDING_CONFIG_FILE = "embedding_config.yaml"
DEFAULT_EMBEDDING_SUBFOLDER = "embedding"

TQDM_BAR_FORMAT = "{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}"


class Backends(str, Enum):
    PYTORCH = "torch"
    TRANSFORMERS = "transformers"
    DATASETS = "datasets"
    TOKENIZERS = "tokenizers"
    SOUNDFILE = "soundfile"
    LIBROSA = "librosa"
    WANDB = "wandb"
    GENSIM = "gensim"
    PILLOW = "PIL"
    JIWER = "jiwer"
    NLTK = "nltk"
    SCIKIT = "sklearn"
    SEQEVAL = "seqeval"

    def __str__(self):
        return str(self.value)


class TaskType(str, Enum):
    AUDIO_CLASSIFICATION = "audio_classification"
    IMAGE2TEXT = "image2text"
    LANGUAGE_MODELING = "language_modeling"
    SEQUENCE_LABELING = "sequence_labeling"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_DETECTION = "text_detection"
    TEXT_GENERATION = "text_generation"


class ConfigType(str, Enum):
    BASE = "base"
    MODEL = "model"
    DATASET = "dataset"
    PREPROCESSOR = "preprocessor"
    EMBEDDING = "embedding"
    TRAINER = "trainer"
    OPTIMIZER = "optimizer"
    CRITERION = "criterion"
    LR_SCHEDULER = "lr_scheduler"
    METRIC = "metric"


class RegistryType(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    PREPROCESSOR = "preprocessor"
    EMBEDDING = "embedding"
    TRAINER = "trainer"
    OPTIMIZER = "optimizer"
    CRITERION = "criterion"
    LR_SCHEDULER = "lr_scheduler"
    METRIC = "metric"


class SplitType(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    VALID = "validation"
    TEST = "test"


class MetricType(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    RECALL = "recall"
    PRECISION = "precision"
    SEQEVAL = "seqeval"
    CER = "cer"
    WER = "wer"
    BLEU = "bleu"


class RepoType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"


class ImageType(str, Enum):
    NUMPY = "numpy"
    PIL = "pil"
    TORCH = "torch"


class ChannelsAxisSide(str, Enum):
    FIRST = "first"
    LAST = "last"
