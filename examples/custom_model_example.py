from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hezar import Model, ModelConfig, register_model


@dataclass
class PerceptronConfig(ModelConfig):
    name = "perceptron"
    input_shape: int = 4
    output_shape: int = 2


@register_model("perceptron", config_class=PerceptronConfig)
class Perceptron(Model):
    """
    A simple single layer network
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.nn = nn.Linear(
            in_features=self.config.input_shape,
            out_features=self.config.output_shape,
        )

    def preprocess(self, raw_inputs, **kwargs):
        inputs_tensor = Tensor(raw_inputs)
        return inputs_tensor

    def forward(self, x: torch.Tensor):
        x = x.reshape(1, -1)
        x = self.nn(x)
        return x

    def post_process(self, model_outputs, **kwargs):
        return model_outputs.numpy()


model = Perceptron(PerceptronConfig())
inputs = [1, 2, 3, 4]
outputs = model.predict(inputs)
print(outputs)
