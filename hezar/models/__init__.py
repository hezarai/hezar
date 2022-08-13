from .base_model import BaseModel
from ..configs import ModelConfig

MODEL_REGISTRY = {}


def register_model(model_name, model_config):
    def register_model_class(cls):
        if model_name in MODEL_REGISTRY:
            raise ValueError(f'Requested model `{model_name}` already exists in the registry!')
        if not issubclass(cls, BaseModel):
            raise ValueError(f'The model class for `{model_name}: {cls.__name__}` must extend `BaseModel`!')
        if not issubclass(model_config, ModelConfig):
            raise ValueError(
                f'The model config for `{model_config}: {model_config.__name__}` must extend `ModelConfig`!')

        MODEL_REGISTRY[model_name] = dict(model=cls, config=model_config)

        return cls

    return register_model_class
