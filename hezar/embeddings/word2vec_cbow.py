from typing import Union, Literal
import os
from dataclasses import dataclass

import gensim

from .embedding import Embedding, EmbeddingConfig

from ..registry import register_embedding
from ..utils import get_local_cache_path


@dataclass
class Word2VecCBOWConfig(EmbeddingConfig):
    name: str = "word2vec_cbow"
    dataset_path: str = None
    save_format: Literal["binary", "text"] = "binary"


@register_embedding("word2vec_cbow", config_class=Word2VecCBOWConfig)
class Word2VecCBOW(Embedding):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def build(self):
        ...

    def __call__(self, inputs, **kwargs):
        ...

    def save(
        self,
        path: Union[str, os.PathLike],
        filename: str = None,
        save_config: bool = True,
        config_filename: str = None,
    ):
        filename = filename or self.filename
        config_filename = config_filename or self.config_filename

        os.makedirs(path, exist_ok=True)
        self.config.save(path, config_filename)

        # TODO
        ...

