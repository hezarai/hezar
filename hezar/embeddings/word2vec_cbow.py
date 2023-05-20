from typing import Union, Literal
from dataclasses import dataclass

import gensim

from .embedding import Embedding, EmbeddingConfig


@dataclass
class Word2VecCBOWConfig(EmbeddingConfig):
    name: str = "word2vec_cbow"
    dataset_path: str = None
    save_format: Literal["binary", "text"] = "binary"


class Word2VecCBOW(Embedding):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def __call__(self, inputs, **kwargs):
        ...
