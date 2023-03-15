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
    criterions_registry,  # noqa
    optimizers_registry,  # noqa
    lr_schedulers_registry,  # noqa
)

__all__ = [
    "build_model",
    "build_dataset",
    "build_preprocessor",
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
    config = config or datasets_registry[name]["config_class"]()
    dataset = datasets_registry[name]["dataset_class"](config, split=split, **kwargs)
    return dataset


def build_criterion(name: str, **kwargs):
    """
    Build the loss function using its registry name.

    Args:
        name (str): Name of the optimizer in the criterions_registry
        **kwargs: Criterion's parameters as keyword arguments

    Returns:
        An nn.Module instance
    """
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
    scheduler = lr_schedulers_registry[name](optimizer, **kwargs)
    return scheduler
