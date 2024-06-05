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
DEFAULT_TRAINER_STATE_FILE = "trainer_state.yaml"
DEFAULT_OPTIMIZER_FILE = "optimizer.pt"
DEFAULT_LR_SCHEDULER_FILE = "lr_scheduler.pt"
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


class ExplicitEnum(str, Enum):
    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        return [x.value for x in cls.__members__.values()]


class Backends(ExplicitEnum):
    """
    All required dependency packages and libraries. Note that the values here must be the exact module names used
    for importing, for example if you set PILLOW the value must be `PIL` not `pillow`, `pil`, etc.
    """

    PYTORCH = "torch"
    TRANSFORMERS = "transformers"
    DATASETS = "datasets"
    TOKENIZERS = "tokenizers"
    ACCELERATE = "accelerate"
    SOUNDFILE = "soundfile"
    LIBROSA = "librosa"
    WANDB = "wandb"
    GENSIM = "gensim"
    PILLOW = "PIL"
    JIWER = "jiwer"
    NLTK = "nltk"
    SCIKIT = "sklearn"
    SEQEVAL = "seqeval"
    ROUGE = "rouge_score"


class TaskType(ExplicitEnum):
    AUDIO_CLASSIFICATION = "audio_classification"
    BACKBONE = "backbone"
    IMAGE2TEXT = "image2text"
    LANGUAGE_MODELING = "language_modeling"
    MASK_FILLING = "mask_filling"
    SEQUENCE_LABELING = "sequence_labeling"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_DETECTION = "text_detection"
    TEXT_GENERATION = "text_generation"


class ConfigType(ExplicitEnum):
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


class RegistryType(ExplicitEnum):
    MODEL = "model"
    DATASET = "dataset"
    PREPROCESSOR = "preprocessor"
    EMBEDDING = "embedding"
    TRAINER = "trainer"
    OPTIMIZER = "optimizer"
    CRITERION = "criterion"
    LR_SCHEDULER = "lr_scheduler"
    METRIC = "metric"


class LossType(ExplicitEnum):
    L1 = "l1"
    NLL = "nll"
    NLL_2D = "nll_2d"
    POISSON_NLL = "poisson_nll"
    GAUSSIAN_NLL = "gaussian_nll"
    MSE = "mse"
    BCE = "bce"
    BCE_WITH_LOGITS = "bce_with_logits"
    CROSS_ENTROPY = "cross_entropy"
    TRIPLE_MARGIN = "triple_margin"
    CTC = "ctc"


class PrecisionType(ExplicitEnum):
    NO = "no"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class OptimizerType(ExplicitEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    SDG = "sdg"


class LRSchedulerType(ExplicitEnum):
    CONSTANT = "constant"
    LAMBDA = "lambda"
    STEP = "step"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    MULTI_STEP = "multi_step"
    ONE_CYCLE = "one_cycle"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CYCLIC = "cyclic"
    SEQUENTIAL = "sequential"
    POLYNOMIAL = "polynomial"
    COSINE_ANEALING = "cosine_anealing"


class SplitType(ExplicitEnum):
    TRAIN = "train"
    EVAL = "eval"
    VALID = "validation"
    TEST = "test"


class MetricType(ExplicitEnum):
    ACCURACY = "accuracy"
    F1 = "f1"
    RECALL = "recall"
    PRECISION = "precision"
    SEQEVAL = "seqeval"
    CER = "cer"
    WER = "wer"
    BLEU = "bleu"
    ROUGE = "rouge"


class RepoType(ExplicitEnum):
    DATASET = "dataset"
    MODEL = "model"


class ImageType(ExplicitEnum):
    NUMPY = "numpy"
    PILLOW = "pillow"
    TORCH = "torch"


class ChannelsAxisSide(ExplicitEnum):
    FIRST = "first"
    LAST = "last"


class PaddingType(ExplicitEnum):
    MAX_LENGTH = "max_length"
    LONGEST = "longest"


class Color(ExplicitEnum):
    HEADER = "\033[95m"
    NORMAL = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\33[3m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GREY = "\33[90m"
