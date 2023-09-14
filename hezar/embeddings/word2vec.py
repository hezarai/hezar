import os
import pprint
from dataclasses import dataclass
from typing import List, Literal, Union

from huggingface_hub import hf_hub_download

from ..constants import HEZAR_CACHE_DIR, Backends
from ..registry import register_embedding
from ..utils import is_backend_available
from .embedding import Embedding, EmbeddingConfig


if is_backend_available(Backends.GENSIM):
    from gensim.models import word2vec

_required_backends = [
    Backends.GENSIM,
]


@dataclass
class Word2VecConfig(EmbeddingConfig):
    name = "word2vec"
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
    required_backends = _required_backends
    vectors_filename = f"{Embedding.filename}.wv.vectors.npy"

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
                    cache_dir=HEZAR_CACHE_DIR,
                    resume_download=True,
                )
                vectors_path = hf_hub_download(
                    pretrained_path,
                    filename=self.vectors_filename,
                    subfolder=self.subfolder,
                    cache_dir=HEZAR_CACHE_DIR,
                    resume_download=True,
                )
            else:
                embedding_path = os.path.join(
                    pretrained_path,
                    self.subfolder,
                    self.filename,
                )
                vectors_path = os.path.join(
                    pretrained_path,
                    self.subfolder,
                    self.vectors_filename,
                )
            if not os.path.isfile(vectors_path):
                raise ValueError(f"Could not load or find vectors file at `{vectors_path}`! "
                                 f"Please make sure it's been downloaded properly!")
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
        subfolder: str = None,
        save_config: bool = True,
        config_filename: str = None,
    ):
        filename = filename or self.filename
        config_filename = config_filename or self.config_filename
        subfolder = subfolder or self.subfolder

        save_dir = os.path.join(path, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        self.config.save(path, config_filename, subfolder=subfolder)

        self.model.save(os.path.join(save_dir, filename))

    def similarity(self, word1: str, word2: str):
        if not isinstance(word1, str) or not isinstance(word2, str):
            raise ValueError(f"`Embedding.similarity()` takes two string objects!\n"
                             f"`word1`: {type(word1)}, `word2`: {type(word2)}")
        similarity = self.word_vectors.similarity(word1, word2)
        return similarity

    def doesnt_match(self, words: List[str]):
        doesnt_match = self.word_vectors.doesnt_match(words)
        return doesnt_match

    def most_similar(self, word: str, top_n: int = 5):
        if not isinstance(word, str):
            raise ValueError(f"`word` must be `str`, got `{type(word)}`!")
        most_similar = self.word_vectors.most_similar(word, topn=top_n)
        most_similar = [{"word": word, "score": f"{score:.4f}"} for word, score in most_similar]
        return pprint.pformat(most_similar)

    def get_normed_vectors(self):
        normed_vectors = self.word_vectors.get_normed_vectors()
        return normed_vectors

    @property
    def word_vectors(self):
        return self.model.wv

    @property
    def vectors(self):
        return self.model.wv.vectors

    @property
    def vocab(self):
        return self.model.wv.key_to_index
