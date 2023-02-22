import importlib
import os

from hezar.registry import datasets_registry

__all__ = [
    "datasets_registry",
    "register_dataset",
    "import_datasets",
]


def register_dataset(dataset_name: str, config_class):
    def register_dataset_class(cls):
        if dataset_name in datasets_registry:
            raise ValueError(f"Requested dataset `{dataset_name}` already exists in the registry!")

        config_class.name = dataset_name
        datasets_registry[dataset_name] = dict(dataset_class=cls, config_class=config_class)

        return cls

    return register_dataset_class


def import_datasets(datasets_dir, namespace):
    for module in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, module)
        if (
            not module.startswith("_")
            and not module.startswith(".")
            and (module.endswith(".py") or os.path.isdir(path))
        ):
            dataset_name = module[: module.find(".py")] if module.endswith(".py") else module
            importlib.import_module(namespace + "." + dataset_name)


# import all datasets in the `datasets` module so that their classes are registered
import_datasets(os.path.dirname(__file__), "hezar.data.datasets")
