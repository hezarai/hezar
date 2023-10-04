from dataclasses import dataclass

from torch import Tensor, nn

from hezar import Model, ModelConfig


@dataclass
class PerceptronConfig(ModelConfig):
    name = "perceptron"
    input_shape: int = 4
    output_shape: int = 2


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

    def forward(self, inputs: list, **kwargs):
        inputs = Tensor(inputs).reshape(1, -1)
        x = self.nn(inputs)
        return x

    def post_process(self, model_outputs, **kwargs):
        return model_outputs.numpy()


model = Perceptron(PerceptronConfig())
inputs = [1, 2, 3, 4]
outputs = model.predict(inputs)
print(outputs)
