from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal

from ..constants import Backends
from ..registry import register_embedding
from ..utils import is_backend_available
from .embedding import Embedding, EmbeddingConfig


if is_backend_available(Backends.GENSIM):
    from gensim.models import fasttext

_required_backends = [
    Backends.GENSIM,
]


@dataclass
class FastTextConfig(EmbeddingConfig):
    """
    Configuration class for FastText embeddings.

    Attributes:
        name (str): Name of the embedding.
        dataset_path (str): Path to the dataset.
        vector_size (int): Size of the word vectors.
        window (int): Window size for context words.
        alpha (float): Learning rate.
        min_count (int): Ignores all words with a total frequency lower than this.
        seed (int): Seed for random number generation.
        workers (int): Number of workers for training.
        min_alpha (float): Minimum learning rate.
        train_algorithm (Literal["skipgram", "cbow"]): Training algorithm, either 'skipgram' or 'cbow'.
        cbow_mean (int): Constant for CBOW. Default is 1.
        epochs (int): Number of training epochs. Default is 5.
    """

    name = "fasttext"
    dataset_path: str = None
    vector_size: int = 300
    window: int = 5
    alpha: float = 0.025
    min_count: int = 1
    seed: int = 1
    workers: int = 3
    min_alpha: float = 0.0001
    train_algorithm: Literal["skipgram", "cbow"] = "skipgram"
    cbow_mean: int = 1
    epochs: int = 5


@register_embedding("fasttext", config_class=FastTextConfig)
class FastText(Embedding):
    """
    FastText embedding class.

    Args:
        config (FastTextConfig): Configuration object.
        embedding_file (str): Path to the embedding file.
        vectors_file (str): Path to the vectors file.
        **kwargs: Additional config parameters given as keyword arguments.
    """

    required_backends = _required_backends

    def __init__(self, config: FastTextConfig, embedding_file: str = None, vectors_file: str = None, **kwargs):
        super().__init__(config, embedding_file=embedding_file, vectors_file=vectors_file, **kwargs)

    def build(self):
        """
        Build the FastText embedding model.

        Returns:
            fasttext.FastText: FastText embedding model.
        """
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

    def from_file(self, embedding_path, vectors_path):
        """
        Load the FastText embedding model from file.

        Args:
            embedding_path (str): Path to the embedding file.
            vectors_path (str): Path to the vectors file.

        Returns:
            fasttext.FastText: Loaded FastText embedding model.
        """
        if not os.path.isfile(vectors_path):
            raise ValueError(
                f"Could not load or find vectors file at `{vectors_path}`! "
                f"Please make sure it's been downloaded properly!"
            )

        embedding_model = fasttext.FastText.load(embedding_path)

        return embedding_model

    def train(
        self,
        dataset: List[str],
        epochs: int = 5,
    ):
        """
        Train the FastText embedding model.

        Args:
            dataset (List[str]): List of sentences for training.
            epochs (int): Number of training epochs.
        """
        self.model.build_vocab(dataset)
        self.model.train(
            dataset,
            epochs=epochs,
            total_examples=self.model.corpus_count,
            total_words=self.model.corpus_total_words,
        )

    def save(
        self,
        path: str | os.PathLike,
        filename: str = None,
        subfolder: str = None,
        save_config: bool = True,
        config_filename: str = None,
    ):
        """
        Save the FastText embedding model to a specified path.

        Args:
            path (str | os.PathLike): Path to save the embedding model.
            filename (str): Name of the embedding file.
            subfolder (str): Subfolder within the path.
            save_config (bool): Whether to save the configuration.
            config_filename (str): Configuration file name.
        """
        filename = filename or self.filename
        config_filename = config_filename or self.config_filename
        subfolder = subfolder or self.subfolder

        save_dir = os.path.join(path, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        self.config.save(path, config_filename, subfolder=subfolder)

        self.model.save(os.path.join(save_dir, filename))

    def similarity(self, word1: str, word2: str):
        """
        Get the similarity between two words.

        Args:
            word1 (str): First word.
            word2 (str): Second word.

        Returns:
            float: Similarity score.
        """
        if not isinstance(word1, str) or not isinstance(word2, str):
            raise ValueError(
                f"`Embedding.similarity()` takes two string objects!\n"
                f"`word1`: {type(word1)}, `word2`: {type(word2)}"
            )
        similarity = self.word_vectors.similarity(word1, word2)
        return similarity

    def doesnt_match(self, words: List[str]):
        """
        Get the word that doesn't match the others in a list.

        Args:
            words (List[str]): List of words.

        Returns:
            str: Word that doesn't match.
        """
        doesnt_match = self.word_vectors.doesnt_match(words)
        return doesnt_match

    def most_similar(self, word: str, top_n: int = 5):
        """
        Get the most similar words to a given word.

        Args:
            word (str): Input word.
            top_n (int): Number of similar words to retrieve.

        Returns:
            List[Dict[str, str | float]]: List of dictionaries containing 'word' and 'score'.
        """
        if not isinstance(word, str):
            raise ValueError(f"`word` must be `str`, got `{type(word)}`!")
        most_similar = self.word_vectors.most_similar(word, topn=top_n)
        most_similar = [{"word": word, "score": f"{score:.4f}"} for word, score in most_similar]
        return most_similar

    def get_normed_vectors(self):
        """
        Get normalized word vectors.
        """
        normed_vectors = self.word_vectors.get_normed_vectors()
        return normed_vectors

    @property
    def word_vectors(self):
        """
        Get word vectors.
        """
        return self.model.wv

    @property
    def vectors(self):
        """
        Get all vectors.
        """
        return self.model.wv.vectors

    @property
    def vocab(self):
        """
        Get vocabulary.
        """
        return self.model.wv.key_to_index
