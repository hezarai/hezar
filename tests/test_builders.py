from hezar.builders import *


def build_models():
    from hezar.registry import models_registry

    for name, module in models_registry.items():
        model = build_model(name, config=module["config_class"](), num_labels=2)
        print(f"Succesfully built `{name}`")


def build_preprocessors():
    from hezar.registry import preprocessors_registry

    for name, module in preprocessors_registry.items():
        model = build_preprocessor(name, config=module["config_class"]())
        print(f"Succesfully built `{name}`")


if __name__ == "__main__":
    build_models()
    build_preprocessors()
