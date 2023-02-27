from hezar.builders import *


def build_models():
    from hezar.registry import models_registry

    for name, module in models_registry.items():
        model = build_model(name, config=module["config_class"]())
        print(f"Succesfully built `{name}`")


def build_datasets():
    from hezar.registry import datasets_registry

    for name, module in datasets_registry.items():
        model = build_dataset(name, config=module["config_class"]())
        print(f"Succesfully built `{name}`")


def build_preprocessors():
    from hezar.registry import preprocessors_registry

    for name, module in preprocessors_registry.items():
        model = build_preprocessor(name, config=module["config_class"]())
        print(f"Succesfully built `{name}`")


if __name__ == "__main__":
    build_models()
    build_datasets()
    build_preprocessors()
