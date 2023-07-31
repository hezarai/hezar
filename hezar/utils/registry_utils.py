__all__ = [
    "list_available_models",
    "list_available_preprocessors",
    "list_available_datasets",
    "list_available_metrics",
    "list_available_trainers",
    "list_available_embeddings",
]


def list_available_models():
    from ..registry import models_registry
    return list(models_registry.keys())


def list_available_preprocessors():
    from ..registry import preprocessors_registry
    return list(preprocessors_registry.keys())


def list_available_datasets():
    from ..registry import datasets_registry
    return list(datasets_registry.keys())


def list_available_metrics():
    from ..registry import metrics_registry
    return list(metrics_registry.keys())


def list_available_trainers():
    from ..registry import trainers_registry
    return list(trainers_registry.keys())


def list_available_embeddings():
    from ..registry import embeddings_registry
    return list(embeddings_registry.keys())
