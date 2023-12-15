from __future__ import annotations

import os
import tempfile
from typing import Dict, List

from huggingface_hub import HfApi, hf_hub_download

from ..builders import build_embedding
from ..configs import EmbeddingConfig
from ..constants import (
    DEFAULT_EMBEDDING_CONFIG_FILE,
    DEFAULT_EMBEDDING_FILE,
    DEFAULT_EMBEDDING_SUBFOLDER,
    HEZAR_CACHE_DIR,
    Backends,
)
from ..utils import Logger, get_lib_version, verify_dependencies


logger = Logger(__name__)

# The below code is a workaround. Gensim's models have this limitation that the models can only be loaded using the same
# gensim & numpy version they were saved with.
REQUIRED_GENSIM_VERSION = "4.3.2"
REQUIRED_NUMPY_VERSION = "1.24"


# Check if the right combo of gensim/numpy versions are installed
def _verify_gensim_installation():
    if (
        not get_lib_version("numpy").startswith(REQUIRED_NUMPY_VERSION)
        or not get_lib_version("gensim").startswith(REQUIRED_GENSIM_VERSION)
    ):
        raise ImportError(
            f"The embeddings module in this version of Hezar, requires a combo of numpy>={REQUIRED_NUMPY_VERSION} and "
            f"gensim=={REQUIRED_GENSIM_VERSION}. Please install them by running: \n"
            f"`pip install numpy~={REQUIRED_NUMPY_VERSION} gensim=={REQUIRED_GENSIM_VERSION}`\n"
            f"and make sure to restart your runtime if you're on a notebook environment!\n"
            f"You can also set `bypass_version_check=True` in the embedding's config so that this error is not raised."
        )


class Embedding:
    """
    Base class for all embeddings.

    Args:
        config: An EmbeddingConfig object to construct the embedding.
        embedding_file (str): Path to the embedding file.
        vectors_file (str): Path to the vectors file.
        **kwargs: Extra embedding config parameters passed as keyword arguments.
    """

    required_backends: List[str | Backends] = []

    filename = DEFAULT_EMBEDDING_FILE
    vectors_filename = f"{filename}.wv.vectors.npy"
    config_filename = DEFAULT_EMBEDDING_CONFIG_FILE
    subfolder = DEFAULT_EMBEDDING_SUBFOLDER

    def __init__(self, config: EmbeddingConfig, embedding_file: str = None, vectors_file: str = None, **kwargs):
        verify_dependencies(self, self.required_backends)  # Check if all the required dependencies are installed
        self.config = config.update(kwargs)
        if not self.config.bypass_version_check:
            _verify_gensim_installation()

        self.config = config.update(kwargs)
        self.model = self.from_file(embedding_file, vectors_file) if embedding_file else self.build()

    def build(self):
        """
        Build the embedding model.
        """
        raise NotImplementedError

    def from_file(self, embedding_path, vectors_path):
        """
        Load the embedding model from file.

        Args:
            embedding_path (str): Path to the embedding file.
            vectors_path (str): Path to the vectors file.
        """
        raise NotImplementedError

    def __call__(self, inputs: str | List[str], **kwargs):
        """
        Get vectors for input words.

        Args:
            inputs (str | List[str]): Input word(s).
            **kwargs: Additional keyword arguments.

        Returns:
            List: List of word vectors.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        vectors = [self.word_vectors[w] for w in inputs]
        return vectors

    def train(self, dataset, epochs):
        """
        Train the embedding model on a dataset.

        Args:
            dataset: The training dataset.
            epochs: Number of training epochs.
        """
        raise NotImplementedError

    def word2index(self, word):
        """
        Get the index of a word in the vocabulary.

        Args:
            word (str): Input word.

        Returns:
            int: Index of the word.
        """
        return self.vocab.get(word, -1)

    def index2word(self, index):
        """
        Get the word corresponding to a given index.

        Args:
            index (int): Input index.

        Returns:
            str: Word corresponding to the index.
        """
        keyed_vocab = {v: k for k, v in self.vocab.items()}
        return keyed_vocab[index]

    def similarity(self, word1: str, word2: str):
        """
        Get the similarity between two words.

        Args:
            word1 (str): First word.
            word2 (str): Second word.
        """
        raise NotImplementedError

    def doesnt_match(self, words: List[str]):
        """
        Get the word that doesn't match the others in a list.

        Args:
            words (List[str]): List of words.
        """
        raise NotImplementedError

    def most_similar(self, word: str, top_n: int = 5):
        """
        Get the most similar words to a given word.

        Args:
            word (str): Input word.
            top_n (int): Number of similar words to retrieve.
        """
        raise NotImplementedError

    def get_normed_vectors(self):
        """
        Get normalized word vectors.
        """
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        config_filename=None,
        embedding_file=None,
        vectors_file=None,
        subfolder=None,
        cache_dir=None,
        **kwargs,
    ) -> "Embedding":
        """
        Load an embedding model from a local or Hugging Face Hub path.

        Args:
            hub_or_local_path: Path to the local directory or the Hugging Face Hub repository.
            config_filename (str): Configuration file name.
            embedding_file (str): Embedding file name.
            vectors_file (str): Vectors file name.
            subfolder (str): Subfolder within the repository.
            cache_dir (str): Path to cache directory
            **kwargs: Additional keyword arguments.

        Returns:
            Embedding: Loaded Embedding object.
        """
        config_filename = config_filename or cls.config_filename
        embedding_file = embedding_file or cls.filename
        vectors_file = vectors_file or cls.vectors_filename
        subfolder = subfolder or cls.subfolder
        cache_dir = cache_dir or HEZAR_CACHE_DIR

        config = EmbeddingConfig.load(
            hub_or_local_path,
            filename=config_filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )

        if os.path.isdir(hub_or_local_path):
            embedding_path = os.path.join(hub_or_local_path, subfolder, embedding_file)
            vectors_path = os.path.join(hub_or_local_path, subfolder, vectors_file)
        else:
            embedding_path = hf_hub_download(
                hub_or_local_path,
                filename=embedding_file,
                subfolder=subfolder,
                cache_dir=cache_dir,
                resume_download=True,
            )
            vectors_path = hf_hub_download(
                hub_or_local_path,
                filename=vectors_file,
                subfolder=subfolder,
                cache_dir=cache_dir,
                resume_download=True,
            )

        embedding = build_embedding(
            config.name,
            config=config,
            embedding_file=embedding_path,
            vectors_file=vectors_path,
            **kwargs,
        )

        return embedding

    def save(
        self,
        path: str | os.PathLike,
        filename: str = None,
        subfolder: str = None,
        save_config: bool = True,
        config_filename: str = None,
    ):
        """
        Save the embedding model to a specified path.

        Args:
            path (str | os.PathLike): Path to save the embedding model.
            filename (str): Name of the embedding file.
            subfolder (str): Subfolder within the path.
            save_config (bool): Whether to save the configuration.
            config_filename (str): Configuration file name.
        """
        raise NotImplementedError

    def push_to_hub(
        self,
        repo_id,
        commit_message=None,
        subfolder=None,
        filename=None,
        vectors_filename=None,
        config_filename=None,
        private=False,
    ):
        """
        Push the embedding model to the Hugging Face Hub.

        Args:
            repo_id: ID of the Hugging Face Hub repository.
            commit_message (str): Commit message.
            subfolder (str): Subfolder within the repository.
            filename (str): Name of the embedding file.
            vectors_filename (str): Name of the vectors file.
            config_filename (str): Configuration file name.
            private (bool): Whether the repository is private.
        """
        subfolder = subfolder or self.subfolder
        filename = filename or self.filename
        vectors_filename = vectors_filename or self.vectors_filename
        config_filename = config_filename or self.config_filename

        api = HfApi()
        # create remote repo
        api.create_repo(repo_id, exist_ok=True)
        # save to tmp and prepare for push
        cache_path = tempfile.mkdtemp()
        # save embedding model file
        embedding_save_dir = os.path.join(cache_path)
        os.makedirs(embedding_save_dir, exist_ok=True)

        if commit_message is None:
            commit_message = "Hezar: Upload embedding and config"

        self.save(embedding_save_dir, filename, subfolder=subfolder, save_config=False)

        self.config.push_to_hub(
            repo_id,
            config_filename,
            subfolder=subfolder,
            repo_type="model",
            private=private,
            commit_message=commit_message,
        )

        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=os.path.join(embedding_save_dir, subfolder, filename),
            repo_type="model",
            path_in_repo=f"{subfolder}/{filename}",
            commit_message=commit_message,
        )
        logger.log_upload_success(
            name=f"{self.__class__.__name__}(name={self.config.name})",
            target_path=f"{os.path.join(repo_id, subfolder, filename)}",
        )

        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=os.path.join(embedding_save_dir, subfolder, vectors_filename),
            repo_type="model",
            path_in_repo=f"{subfolder}/{vectors_filename}",
            commit_message=commit_message,
        )
        logger.log_upload_success(
            name=f"`{self.__class__.__name__}(name={self.config.name})`",
            target_path=f"`{os.path.join(repo_id, subfolder, vectors_filename)}`",
        )

    def torch_embedding(self):
        """
        Convert the embedding model to a PyTorch Embedding layer.

        Returns:
            torch.nn.Embedding: PyTorch Embedding layer.
        """
        import torch

        weights = torch.FloatTensor(self.vectors)
        embedding_layer = torch.nn.Embedding.from_pretrained(weights)
        return embedding_layer

    @property
    def word_vectors(self):
        """
        Get key:value pairs of word:vector.
        """
        raise NotImplementedError

    @property
    def vectors(self):
        """
        Get the all vectors array/tensor.
        """
        raise NotImplementedError

    @property
    def vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary.
        """
        raise NotImplementedError
