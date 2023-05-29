import os
from dataclasses import dataclass
from typing import List, Literal, Union

from gensim.models import fasttext
from huggingface_hub import hf_hub_download

from ..constants import HEZAR_CACHE_DIR
from ..registry import register_embedding
from .embedding import Embedding, EmbeddingConfig


@dataclass
class FastTextConfig(EmbeddingConfig):
    name: str = "fasttext"
    dataset_path: str = None
    vector_size: int = 300
    window: int = 5
    alpha: float = 0.025
    min_count: int = 1
    seed: int = 1
    workers: int = 3
    min_alpha: float = 0.0001
    train_algorithm: Literal["skipgram", "cbow"] = "skipgram"
    cbow_mean = 1
    epochs = 5
    pretrained_path: str = None


@register_embedding("fasttext", config_class=FastTextConfig)
class FastText(Embedding):
    filename = "embedding.model"
    array_filename = "embedding.model.wv.vectors_ngrams.npy"

    def __init__(self, config: FastTextConfig, **kwargs):
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
                    cache_dir=HEZAR_CACHE_DIR,
                )

            else:
                embedding_path = os.path.join(
                    pretrained_path,
                    self.subfolder,
                    self.filename,
                )
            embedding_model = fasttext.FastText.load(embedding_path)
        else:
            embedding_model = fasttext.FastText(
                vector_size=self.config.vector_size,
                window=self.config.window,
                sg=1 if self.config.train_algorithm == "skipgram" else 0,
                workers=self.config.workers,
                alpha=self.config.alpha,
                min_alpha=self.config.min_alpha,
                min_count=self.config.min_count,
            )

        return embedding_model

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

    @property
    def word_vectors(self):
        return self.model.wv

    @property
    def vectors(self):
        return self.model.wv.vectors

    @property
    def vocab(self):
        return self.model.wv.key_to_index
