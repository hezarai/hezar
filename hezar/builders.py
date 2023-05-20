r"""
Builder functions are used to create an instance of a module e.g, models, preprocessors, etc. without having to import
their corresponding classes manually. These builders use modules' registries to do so. Every builder gets a name and
optional config or config kwargs to build the object.

Examples:

    >>> from hezar.builders import build_model
    >>> model = build_model('distilbert_text_classification', id2label={0: 'negative', 1: 'positive'})
    >>> print(model)

"""

from .registry import (  # noqa
    models_registry,  # noqa
    preprocessors_registry,  # noqa
    datasets_registry,  # noqa
    embeddings_registry, # noqa
    criterions_registry,  # noqa
    optimizers_registry,  # noqa
    lr_schedulers_registry,  # noqa
)

__all__ = [
    "build_model",
    "build_dataset",
    "build_preprocessor",
    "build_embedding",
    "build_optimizer",
    "build_criterion",
    "build_scheduler",
]


def build_model(name: str, config=None, **kwargs):
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
    if name not in models_registry:
        raise ValueError(f"Unknown model name: `{name}`!\n"
                         f"Available model names: {list(models_registry.keys())}")
    config = config or models_registry[name]["config_class"]()
    model = models_registry[name]["model_class"](config, **kwargs)
    return model


def build_preprocessor(name: str, config=None, **kwargs):
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
    if name not in preprocessors_registry:
        raise ValueError(f"Unknown preprocessor name: `{name}`!\n"
                         f"Available preprocessor names: {list(preprocessors_registry.keys())}")
    config = config or preprocessors_registry[name]["config_class"]()
    preprocessor = preprocessors_registry[name]["preprocessor_class"](config, **kwargs)
    return preprocessor


def build_dataset(name: str, config=None, split=None, **kwargs):
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
    if name not in datasets_registry:
        raise ValueError(f"Unknown dataset name: `{name}`!\n"
                         f"Available dataset names: {list(datasets_registry.keys())}")
    config = config or datasets_registry[name]["config_class"]()
    dataset = datasets_registry[name]["dataset_class"](config, split=split, **kwargs)
    return dataset


def build_embedding(name: str, config=None, **kwargs):
    """
    Build the embedding using its registry name. If config is None then the embedding is built using the
    default config.

    Args:
        name (str): Name of the embedding in the embeddings' registry
        config (EmbeddingConfig): An EmbeddingConfig instance
        **kwargs: Extra config parameters that are loaded to the embedding

    Returns:
        A Dataset instance
    """
    if name not in embeddings_registry:
        raise ValueError(f"Unknown embedding name: `{name}`!\n"
                         f"Available embedding names: {list(embeddings_registry.keys())}")
    config = config or embeddings_registry[name]["config_class"]()
    embedding = embeddings_registry[name]["embedding_class"](config, **kwargs)
    return embedding


def build_criterion(name: str, **kwargs):
    """
    Build the loss function using its registry name.

    Args:
        name (str): Name of the optimizer in the criterions_registry
        **kwargs: Criterion's parameters as keyword arguments

    Returns:
        An nn.Module instance
    """
    if name not in criterions_registry:
        raise ValueError(f"Unknown criterion name: `{name}`!\n"
                         f"Available criterion names: {list(criterions_registry.keys())}")
    criterion = criterions_registry[name](**kwargs)
    return criterion


def build_optimizer(name: str, params, **kwargs):
    """
    Build the optimizer using its registry name.

    Args:
        name (str): Name of the optimizer in the optimizers_registry
        params (Iterator[nn.Parameter]): Model parameters
        **kwargs: Optimizer parameters as keyword arguments

    Returns:
        An optim.Optimizer instance
    """
    if name not in optimizers_registry:
        raise ValueError(f"Unknown optimizer name: `{name}`!\n"
                         f"Available optimizer names: {list(optimizers_registry.keys())}")
    optimizer = optimizers_registry[name](params, **kwargs)
    return optimizer


def build_scheduler(name: str, optimizer, **kwargs):
    """
    Build the LR scheduler using its registry name.

    Args:
        name (str): Name of the optimizer in the lr_schedulers_registry
        optimizer (optim.Optimizer): The optimizer
        **kwargs: Scheduler parameters as keyword arguments

    Returns:
        An optim.lr_scheduler._LRScheduler instance
    """
    if name not in lr_schedulers_registry:
        raise ValueError(f"Unknown LR scheduler name: `{name}`!\n"
                         f"Available LR scheduler names: {list(lr_schedulers_registry.keys())}")
    scheduler = lr_schedulers_registry[name](optimizer, **kwargs)
    return scheduler
