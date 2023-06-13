r"""
Hezar uses a registry system in a way that for any core module like model, dataset, etc. there is an entry in its
specific registry. These registries are simple python dictionaries that map a module's name to its class and its config
class. These registries are initialized here and filled automatically when you import hezar or a registry itself.

Examples:
    >>> # read models registry
    >>> from hezar.registry import models_registry
    >>> print(models_registry)
    {'distilbert_lm': {'model_class': <class 'hezar.models.language_modeling.distilbert.distilbert_lm.DistilBertLM'>,
    'config_class': <class 'hezar.models.language_modeling.distilbert.distilbert_lm_config.DistilBertLMConfig'>}}

    >>> # add a model class to models_registry
    >>> from hezar import Model, register_model
    >>> @register_model(name="my_awesome_model", config_class=MyAwesomeModelConfig)
    >>> class MyAwesomeModel(Model):
    ...    def __init__(config: MyAwesomeModelConfig):
    ...        ...

Keep in mind that registries usually don't need to be used directly. There is a bunch of functions to build modules
using a module's registry name in `hezar.builders` module. See the file `builders.py` for more info.

Note: In case of adding a new registry container, make sure to add to `__all__` below!
"""

from torch import nn, optim

from .utils import get_logger


__all__ = [
    "register_model",
    "register_preprocessor",
    "register_dataset",
    "register_embedding",
]

logger = get_logger(__name__)

models_registry = {}
preprocessors_registry = {}
datasets_registry = {}
embeddings_registry = {}
trainers_registry = {}

criterions_registry = {
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "ctc": nn.CTCLoss,
}
optimizers_registry = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}
lr_schedulers_registry = {
    "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "cosine_lr": optim.lr_scheduler.CosineAnnealingLR,
}


def register_model(model_name: str, config_class):
    """
    A class decorator that adds the model class and the config class to the `models_registry`

    Args:
        model_name: Model's registry name e.g, `bert_sequence_labeling`
        config_class: Model's config class e.g, `BertSequenceLabelingConfig`. This parameter must be the config class
            itself not a config instance!

    Returns:
         The class itself
    """
    def register(cls):
        if model_name in models_registry:
            logger.warning(f"Model `{model_name}` is already registered. Overwriting...")

        if config_class.name != model_name:
            raise ValueError(f"`model_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"model_name: {model_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        models_registry[model_name] = {"model_class": cls, "config_class": config_class}

        return cls

    return register


def register_dataset(dataset_name: str, config_class):
    """
    A class decorator that adds the dataset class and the config class to the `datasets_registry`

    Args:
        dataset_name: Dataset's registry name e.g, `text_classification`.
        config_class: Dataset's config class e.g, `TextClassificationDatasetConfig`. This parameter must be the config
            class itself not a config instance!

    Returns:
         The class itself
    """
    def register(cls):
        if dataset_name in datasets_registry:
            logger.warning(f"Dataset `{dataset_name}` is already registered. Overwriting...")

        if config_class.name != dataset_name:
            raise ValueError(f"`dataset_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"dataset_name: {dataset_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        datasets_registry[dataset_name] = {"dataset_class": cls, "config_class": config_class}

        return cls

    return register


def register_preprocessor(preprocessor_name: str, config_class):
    """
    A class decorator that adds the preprocessor class and the config class to the `preprocessors_registry`

    Args:
        preprocessor_name: Preprocessor's registry name e.g, `bpe_tokenizer`.
        config_class: Preprocessor's config class e.g, BPEConfig. This parameter must be the config
            class itself not a config instance!

    Returns:
         The class itself
    """
    def register(cls):
        if preprocessor_name in preprocessors_registry:
            logger.warning(f"Preprocessor `{preprocessor_name}` is already registered. Overwriting...")

        if config_class.name != preprocessor_name:
            raise ValueError(f"`preprocessor_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"preprocessor_name: {preprocessor_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        preprocessors_registry[preprocessor_name] = {"preprocessor_class": cls, "config_class": config_class}

        return cls

    return register


def register_embedding(embedding_name: str, config_class):
    """
    A class decorator that adds the embedding class and the config class to the `embeddings_registry`

    Args:
        embedding_name: Embedding's registry name e.g, `word2vec_cbow`.
        config_class: Embedding's config class e.g, Word2VecCBOWConfig. This parameter must be the config
            class itself not a config instance!

    Returns:
         The class itself
    """
    def register(cls):
        if embedding_name in embeddings_registry:
            logger.warning(f"Embedding `{embedding_name}` is already registered. Overwriting...")

        if config_class.name != embedding_name:
            raise ValueError(f"`embedding_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"embedding_name: {embedding_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        embeddings_registry[embedding_name] = {"embedding_class": cls, "config_class": config_class}

        return cls

    return register


def register_trainer(trainer_name: str, config_class):
    """
    A class decorator that adds the Trainer class and the config class to the `trainers_registry`

    Args:
        trainer_name: Trainer's registry name e.g, `text_classification_trainer`
        config_class: Trainer's config class e.g, `TextClassificationTrainerConfig`.
            This parameter must be the config class itself not a config instance!

    Returns:
         The class itself
    """
    def register(cls):
        if trainer_name in trainers_registry:
            logger.warning(f"Trainer `{trainer_name}` is already registered. Overwriting...")

        if config_class.name != trainer_name:
            raise ValueError(f"`trainer_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"trainer_name: {trainer_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        trainers_registry[trainer_name] = {"trainer_class": cls, "config_class": config_class}

        return cls

    return register
