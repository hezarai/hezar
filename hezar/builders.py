r"""
Builder functions are used to create an instance of a module e.g, models, preprocessors, etc. without having to import
their corresponding classes manually. These builders use modules' registries to do so. Every builder gets a name and
optional config or config kwargs to build the object.

Examples:

    >>> from hezar.builders import build_model
    >>> model = build_model('distilbert_text_classification', id2label={0: 'negative', 1: 'positive'})
    >>> print(model)

"""
from typing import Optional

from .configs import (
    DatasetConfig,
    EmbeddingConfig,
    MetricConfig,
    ModelConfig,
    PreprocessorConfig,
)
from .constants import SplitType
from .registry import (
    datasets_registry,
    embeddings_registry,
    metrics_registry,
    models_registry,
    preprocessors_registry,
)


__all__ = [
    "build_model",
    "build_dataset",
    "build_preprocessor",
    "build_embedding",
    "build_metric",
]


def build_model(name: str, config: Optional[ModelConfig] = None, **kwargs):
    """
    Build the model using its registry name. If config is None then the model is built using the default config. Notice
    that this function only builds the model and does not perform any weights loading/initialization unless these
    actions are done in the model's :func:`__init__` .

    Args:
        name (str): name of the model in the models' registry
        config (ModelConfig): a ModelConfig instance
        **kwargs: extra config parameters that are loaded to the model

    Returns:
        A Model instance
    """
    from .utils import list_available_models

    available_models = list_available_models()
    if name not in available_models:
        raise ValueError(f"Unknown model name: `{name}`!\nAvailable model names: {available_models}")
    config = config or models_registry[name].config_class()
    model = models_registry[name].module_class(config, **kwargs)
    return model


def build_preprocessor(name: str, config: Optional[PreprocessorConfig] = None, **kwargs):
    """
    Build the preprocessor using its registry name. If config is None then the preprocessor is built using the
    default config.

    Args:
        name (str): name of the preprocessor in the preprocessors' registry
        config (PreprocessorConfig): a PreprocessorConfig instance
        **kwargs: extra config parameters that are loaded to the preprocessor

    Returns:
        A Preprocessor instance
    """
    from .utils import list_available_preprocessors

    available_preprocessors = list_available_preprocessors()
    if name not in preprocessors_registry:
        raise ValueError(
            f"Unknown preprocessor name: `{name}`!\nAvailable preprocessor names: {available_preprocessors}"
        )
    config = config or preprocessors_registry[name].config_class()
    preprocessor = preprocessors_registry[name].module_class(config, **kwargs)
    return preprocessor


def build_dataset(name: str, config: Optional[DatasetConfig] = None, split: SplitType = None, **kwargs):
    """
    Build the dataset using its registry name. If config is None then the dataset is built using the
    default config.

    Args:
        name (str): name of the dataset in the datasets' registry
        config (DatasetConfig): a DatasetConfig instance
        split (str): Dataset split to load
        **kwargs: extra config parameters that are loaded to the dataset

    Returns:
        A Dataset instance
    """
    from .utils import list_available_datasets

    available_datasets = list_available_datasets()
    if name not in available_datasets:
        raise ValueError(
            f"Unknown dataset name: `{name}`!\nAvailable dataset names: {available_datasets}"
        )
    config = config or datasets_registry[name].config_class()
    dataset = datasets_registry[name].module_class(config, split, **kwargs)
    return dataset


def build_embedding(name: str, config: Optional[EmbeddingConfig] = None, **kwargs):
    """
    Build the embedding using its registry name. If config is None then the embedding is built using the
    default config.

    Args:
        name (str): Name of the embedding in the embeddings' registry
        config (EmbeddingConfig): An EmbeddingConfig instance
        **kwargs: Extra config parameters that are loaded to the embedding

    Returns:
        A Embedding instance
    """
    from .utils import list_available_embeddings

    available_embeddings = list_available_embeddings()
    if name not in available_embeddings:
        raise ValueError(
            f"Unknown embedding name: `{name}`!\nAvailable embedding names: {available_embeddings}"
        )
    config = config or embeddings_registry[name].config_class()
    embedding = embeddings_registry[name].module_class(config, **kwargs)
    return embedding


def build_metric(name: str, config: Optional[MetricConfig] = None, **kwargs):
    """
    Build the metric using its registry name. If config is None then the metric is built using the
    default config.

    Args:
        name (str): Name of the metric in the metrics' registry
        config (MetricConfig): A MetricConfig instance
        **kwargs: Extra config parameters that are loaded to the metric

    Returns:
        A Metric instance
    """
    from .utils import list_available_metrics

    available_metrics = list_available_metrics()
    if name not in available_metrics:
        raise ValueError(f"Unknown metric name: `{name}`!\nAvailable metric names: {available_metrics}")
    config = config or metrics_registry[name].config_class()
    metric = metrics_registry[name].module_class(config, **kwargs)
    return metric
