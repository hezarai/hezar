from hezar.configs import ModelConfig


class MLPConfig(ModelConfig):
    name: str = 'mlp'
    input_shape: int = 4
    output_shape: int = 2
