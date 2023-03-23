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
    >>> from hezar.registry import register_model
    >>>
    >>> @register_model(name="my_awesome_model", config_class=MyAwesomeModelConfig)
    >>> class MyAwesomeModel(Model):
    >>>     def __init__(config: MyAwesomeModelConfig):
    >>>         ...

Keep in mind that registries usually don't need to be used directly. There is a bunch of functions to build modules
using a module's registry name in `hezar.builders` module. See the file `builders.py` for more info.
"""

from torch import nn, optim


__all__ = [
    "register_model",
    "register_preprocessor",
    "register_dataset",
]

models_registry = {}
preprocessors_registry = {}
datasets_registry = {}
criterions_registry = {
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "ctc": nn.CTCLoss,
}
optimizers_registry = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
lr_schedulers_registry = {
    "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "cosine_lr": optim.lr_scheduler.CosineAnnealingLR,
}


def register_model(model_name: str, config_class):
    def register_model_class(cls):
        if model_name in models_registry:
            raise ValueError(f"Requested model `{model_name}` already exists in the registry!")

        if config_class.name != model_name:
            raise ValueError(f"`model_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"model_name: {model_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        models_registry[model_name] = {"model_class": cls, "config_class": config_class}

        return cls

    return register_model_class


def register_dataset(dataset_name: str, config_class):
    def register_dataset_class(cls):
        if dataset_name in datasets_registry:
            raise ValueError(f"Requested dataset `{dataset_name}` already exists in the registry!")

        if config_class.name != dataset_name:
            raise ValueError(f"`dataset_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"dataset_name: {dataset_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        datasets_registry[dataset_name] = {"dataset_class": cls, "config_class": config_class}

        return cls

    return register_dataset_class


def register_preprocessor(preprocessor_name: str, config_class):
    def register_preprocessor_class(cls):
        if preprocessor_name in preprocessors_registry:
            raise ValueError(f"Requested preprocessor `{preprocessor_name}` already exists in the registry!")

        if config_class.name != preprocessor_name:
            raise ValueError(f"`preprocessor_name` and `config.name` are not compatible for `{cls.__name__}`\n"
                             f"preprocessor_name: {preprocessor_name}\n"
                             f"{config_class.__name__}.name: {config_class.name}")
        preprocessors_registry[preprocessor_name] = {"preprocessor_class": cls, "config_class": config_class}

        return cls

    return register_preprocessor_class
