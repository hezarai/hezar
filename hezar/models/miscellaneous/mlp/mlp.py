from torch import Tensor, nn

from ....models import Model
from ....registry import register_model
from .mlp_config import MLPConfig


@register_model("mlp", config_class=MLPConfig)
class MLP(Model):
    """
    A simple MLP for testing some core features of the base class :class:`hezar.Model`
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.nn = nn.Linear(in_features=self.config.input_shape, out_features=self.config.output_shape)

    def forward(self, inputs: list, **kwargs):
        inputs = Tensor(inputs).reshape(1, -1)
        x = self.nn(inputs)
        return x

    def post_process(self, inputs: Tensor, **kwargs):
        return inputs.argmax(1).item()
