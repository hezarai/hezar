from typing import List

from torch import nn, Tensor

from hezar.models import Model, register_model

from .mlp_config import MLPConfig


@register_model('mlp', config_class=MLPConfig)
class MLP(Model):
    """
    A simple MLP for out tests
    """
    def __init__(self, config):
        super().__init__(config)
        self.nn = nn.Linear(in_features=self.config.input_shape, out_features=self.config.output_shape)

    def forward(self, inputs: Tensor, **kwargs):
        x = self.nn(inputs)
        return x

    def preprocess(self, inputs: List[int], **kwargs):
        inputs = Tensor(inputs).reshape(1, -1)
        return inputs

    def postprocess(self, inputs: Tensor, **kwargs):
        return inputs.max(1).values.item()
