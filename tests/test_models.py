from hezar.builders import build_model
from hezar.registry import models_registry


def build_models():
    for model_name, model in models_registry.items():
        model = build_model(model_name, config=model["config_class"]())
        print(f"Succesfully built `{model_name}`")


if __name__ == '__main__':
    build_models()
