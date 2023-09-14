from ..constants import RegistryType
from .common_utils import snake_case


__all__ = [
    "list_available_models",
    "list_available_preprocessors",
    "list_available_datasets",
    "list_available_metrics",
    "list_available_trainers",
    "list_available_embeddings",
    "get_registry_point",
    "get_module_class",
    "get_module_config_class",
    "get_registry_key_by_module_class",
]


def list_available_models():
    from ..registry import models_registry
    return sorted(models_registry.keys())


def list_available_preprocessors():
    from ..registry import preprocessors_registry
    return sorted(preprocessors_registry.keys())


def list_available_datasets():
    from ..registry import datasets_registry
    return sorted(datasets_registry.keys())


def list_available_metrics():
    from ..registry import metrics_registry
    return sorted(metrics_registry.keys())


def list_available_trainers():
    from ..registry import trainers_registry
    return sorted(trainers_registry.keys())


def list_available_embeddings():
    from ..registry import embeddings_registry
    return sorted(embeddings_registry.keys())


def _get_registry_from_type(registry_type: RegistryType):
    if registry_type == RegistryType.MODEL:
        from ..registry import models_registry  # noqa
        registry = models_registry

    elif registry_type == RegistryType.PREPROCESSOR:
        from ..registry import preprocessors_registry  # noqa
        registry = preprocessors_registry

    elif registry_type == RegistryType.DATASET:
        from ..registry import datasets_registry  # noqa
        registry = datasets_registry

    elif registry_type == RegistryType.EMBEDDING:
        from ..registry import embeddings_registry  # noqa
        registry = embeddings_registry

    elif registry_type == RegistryType.TRAINER:
        from ..registry import trainers_registry  # noqa
        registry = trainers_registry

    elif registry_type == RegistryType.METRIC:
        from ..registry import metrics_registry  # noqa
        registry = metrics_registry

    else:
        raise ValueError(f"Invalid `registry_type`: {registry_type}!")

    return registry


def get_registry_point(registry_key: str, registry_type: RegistryType):
    """
    Get the registry item by registry key name in a specific registry

    Args:
        registry_key: Module's name in the registry
        registry_type: Module's registry container type

    Returns:
        A Registry object
    """
    registry = _get_registry_from_type(registry_type)

    registry = registry[registry_key]
    return registry


def get_module_config_class(name: str, registry_type: RegistryType):
    """
    Get the config class for a given module based on its registry name.

    Args:
        name (str): Module's registry name
        registry_type (str): Registry type

    Returns:
        A class of type :class:`hezar.Config`
    """
    registry = _get_registry_from_type(registry_type)

    config_cls = registry[name].config_class
    return config_cls


def get_module_class(name: str, registry_type: RegistryType):
    """
    Get module class based on registry name

    Args:
        name: Module's key name in its registry
        registry_type: Type of the module e.g, model, dataset, preprocessor, embedding, etc

    Returns:
        A class corresponding to the given module
    """
    registry = _get_registry_from_type(registry_type)

    name = snake_case(name)
    module_cls = registry[name].module_class
    return module_cls


def get_registry_key_by_module_class(module_class: type, registry_type: RegistryType):
    """
    Given the module class, return the registry key if exists

    Args:
        module_class: The module class (raw class, not the name or object)
        registry_type: The registry type

    Returns:
        The corresponding key for the class in its registry
    """
    registry = _get_registry_from_type(registry_type)
    key_values = {v.module_class.__name__: k for k, v in registry.items()}
    module_class_name = module_class.__name__
    if module_class_name not in key_values:
        raise KeyError(
            f"The requested {registry_type} class `{module_class_name}` does not exist "
            f"in the {registry_type}s registry!"
        )
    return key_values[module_class.__name__]
