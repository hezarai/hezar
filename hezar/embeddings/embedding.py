from typing import Union, List

from ..configs import EmbeddingConfig
from ..constants import (
    DEFAULT_EMBEDDING_FILE,
    DEFAULT_EMBEDDING_CONFIG_FILE,
    DEFAULT_EMBEDDING_SUBFOLDER,
)


class Embedding:
    """
    Base class for all embeddings.
    """
    filename = DEFAULT_EMBEDDING_FILE
    config_filename = DEFAULT_EMBEDDING_CONFIG_FILE
    subfolder = DEFAULT_EMBEDDING_SUBFOLDER

    def __init__(self, config: EmbeddingConfig, **kwargs):
        self.config = config.update(kwargs)

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        # TODO
        ...

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        filename=None,
        config_filename=None,
        subfolder=None,
        **kwargs,
    ):
        filename = filename or cls.filename
        config_filename = config_filename or cls.config_filename
        subfolder = subfolder or cls.subfolder

        config = EmbeddingConfig.load(hub_or_local_path, filename=config_filename, subfolder=subfolder, **kwargs)


