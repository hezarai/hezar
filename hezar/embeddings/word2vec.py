from typing import List, Union, Literal
import os
from dataclasses import dataclass

from gensim.models import word2vec
from huggingface_hub import hf_hub_download

from .embedding import Embedding, EmbeddingConfig

from ..registry import register_embedding


@dataclass
class Word2VecConfig(EmbeddingConfig):
    name: str = "word2vec"
    dataset_path: str = None
    vector_size: int = 300
    window: int = 5
    alpha: float = 0.025
    min_count: int = 1
    seed: int = 1
    workers: int = 3
    min_alpha: float = 0.0001
    cbow_mean = 1
    epochs = 5
    train_algorithm: Literal["skipgram", "cbow"] = "skipgram"
    save_format: Literal["binary", "text"] = "binary"
    pretrained_path: str = None


@register_embedding("word2vec", config_class=Word2VecConfig)
class Word2Vec(Embedding):
    def __init__(self, config: Word2VecConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = self.build()

    def build(self):
        pretrained_path = self.config.get("pretrained_path")
        if pretrained_path:
            if not os.path.isdir(pretrained_path):
                embedding_path = hf_hub_download(
                    pretrained_path,
                    filename=self.filename,
                    subfolder=self.subfolder,
                )

            else:
                embedding_path = os.path.join(
                    pretrained_path,
                    self.subfolder,
                    self.filename,
                )
            embedding_model = word2vec.Word2Vec.load(embedding_path)
        else:
            embedding_model = word2vec.Word2Vec(
                vector_size=self.config.vector_size,
                window=self.config.window,
                sg=1 if self.config.train_algorithm == "skipgram" else 0,
                workers=self.config.workers,
                alpha=self.config.alpha,
                min_alpha=self.config.min_alpha,
                min_count=self.config.min_count,
            )

        return embedding_model

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        vectors = [self.model.wv[w] for w in inputs]
        return vectors

    def train(
        self,
        dataset: List[str],
        epochs: int = 5,
    ):
        self.model.build_vocab(dataset)
        self.model.train(
            dataset,
            epochs=epochs,
            total_examples=self.model.corpus_count,
            total_words=self.model.corpus_total_words,
        )

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

        self.model.save(os.path.join(path, filename))
