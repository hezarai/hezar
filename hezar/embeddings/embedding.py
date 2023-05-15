from typing import Union, List

from ..configs import EmbeddingConfig


class Embedding:
    def __init__(self, config: EmbeddingConfig, **kwargs):
        self.config = config.update(kwargs)

    def __call__(self, inputs: Union[str, List[str]]):
        if isinstance(inputs, str):
            inputs = [inputs]
        # TODO
        ...
