r"""
Hezar uses a registry system in a way that for any core module like model, dataset, etc. there is an entry in its
specific registry. These registries are simple python dictionaries that map a module's name to its class and its config
class. These registries are initialized here and filled automatically when you import hezar or a registry itself.

Examples:
    >>> # read models registry
    >>> from hezar.registry import models_registry
    >>> print(models_registry)
    {'distilbert_mask_filling': {'module_class': <class 'hezar.models.mask_filling.distilbert.distilbert_mask_filling.DistilBertMaskFilling'>,
    'config_class': <class 'hezar.models.mask_filling.distilbert.distilbert_mask_filling_config.DistilBertMaskFillingConfig'>},
    'description': 'Optional model description here...'}

    >>> # add a model class to models_registry
    >>> from hezar.models import Model, register_model
    >>> @register_model(name="my_awesome_model", config_class=MyAwesomeModelConfig, description="My Awesome Model!")
    >>> class MyAwesomeModel(Model):
    ...    def __init__(config: MyAwesomeModelConfig):
    ...        ...

Keep in mind that registries usually don't need to be used directly. There is a bunch of functions to build modules
using a module's registry name in `hezar.builders` module. See the file `builders.py` for more info.

Note: In case of adding a new registry container, make sure to add to `__all__` below!
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Type


if TYPE_CHECKING:
    from .configs import (
        Config,
        DatasetConfig,
        EmbeddingConfig,
        MetricConfig,
        ModelConfig,
        PreprocessorConfig,
    )

from .utils import Logger


__all__ = [
    "register_model",
    "register_preprocessor",
    "register_dataset",
    "register_embedding",
    "register_metric",
    "Registry",
    "models_registry",
    "preprocessors_registry",
    "datasets_registry",
    "embeddings_registry",
    "metrics_registry",
]

logger = Logger(__name__)


@dataclass
class Registry:
    module_class: type
    config_class: type = None
    description: Optional[str] = None


models_registry: Dict[str, Registry] = {}
preprocessors_registry: Dict[str, Registry] = {}
datasets_registry: Dict[str, Registry] = {}
embeddings_registry: Dict[str, Registry] = {}
metrics_registry: Dict[str, Registry] = {}


def _register_module(
    cls: Type,
    registry: Dict[str, Registry],
    module_name: str,
    config_class: Type["Config"],
    description: str = None
):
    """
    Add module to the registry.

    Args:
        cls: The module class
        registry: Module's registry container
        module_name: Module's registry name (key)
        config_class: Module's config class
        description: Optional description for the module
    """
    if module_name in registry:
        logger.warning(f"`{module_name}` is already registered. Overwriting...")

    if config_class.name != module_name:
        raise ValueError(
            f"Module's registry name and `config.name` are not compatible for `{cls.__name__}`\n"
            f"Registry name: {module_name}\n"
            f"{config_class.__name__}.name: {config_class.name}"
        )
    registry[module_name] = Registry(module_class=cls, config_class=config_class, description=description)


def register_model(model_name: str, config_class: Type["ModelConfig"], description: str = None):
    """
    A class decorator that adds the model class and the config class to the `models_registry`

    Args:
        model_name: Model's registry name e.g, `bert_sequence_labeling`
        config_class: Model's config class e.g, `BertSequenceLabelingConfig`. This parameter must be the config class
            itself not a config instance!
        description: Optional model description
    """

    def register(cls):
        _register_module(cls, models_registry, model_name, config_class, description)
        return cls

    return register


def register_dataset(dataset_name: str, config_class: Type["DatasetConfig"], description: str = None):
    """
    A class decorator that adds the dataset class and the config class to the `datasets_registry`

    Args:
        dataset_name: Dataset's registry name e.g, `text_classification`.
        config_class: Dataset's config class e.g, `TextClassificationDatasetConfig`. This parameter must be the config
            class itself not a config instance!
        description: Optional dataset description
    """

    def register(cls):
        _register_module(cls, datasets_registry, dataset_name, config_class, description)
        return cls

    return register


def register_preprocessor(preprocessor_name: str, config_class: Type["PreprocessorConfig"], description: str = None):
    """
    A class decorator that adds the preprocessor class and the config class to the `preprocessors_registry`

    Args:
        preprocessor_name: Preprocessor's registry name e.g, `bpe_tokenizer`.
        config_class: Preprocessor's config class e.g, BPEConfig. This parameter must be the config
            class itself not a config instance!
        description: Optional preprocessor description
    """

    def register(cls):
        _register_module(cls, preprocessors_registry, preprocessor_name, config_class, description)
        return cls

    return register


def register_embedding(embedding_name: str, config_class: Type["EmbeddingConfig"], description: str = None):
    """
    A class decorator that adds the embedding class and the config class to the `embeddings_registry`

    Args:
        embedding_name: Embedding's registry name e.g, `word2vec_cbow`.
        config_class: Embedding's config class e.g, Word2VecCBOWConfig. This parameter must be the config
            class itself not a config instance!
        description: Optional embedding description
    """

    def register(cls):
        _register_module(cls, embeddings_registry, embedding_name, config_class, description)
        return cls

    return register


def register_metric(metric_name: str, config_class: Type["MetricConfig"], description: str = None):
    """
    A class decorator that adds the metric class and the config class to the `metrics_registry`

    Args:
        metric_name: Metric registry name e.g, `f1`
        config_class: Metric config class
        description: Optional metric description
    """

    def register(cls):
        _register_module(cls, metrics_registry, metric_name, config_class, description)
        return cls

    return register
