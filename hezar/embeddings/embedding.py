import os
from typing import Union, List

from huggingface_hub import HfApi, upload_file

from ..configs import EmbeddingConfig
from ..constants import (
    DEFAULT_EMBEDDING_FILE,
    DEFAULT_EMBEDDING_CONFIG_FILE,
    DEFAULT_EMBEDDING_SUBFOLDER,
)
from ..builders import build_embedding
from ..utils import resolve_pretrained_path, get_local_cache_path, get_logger

logger = get_logger(__name__)


class Embedding:
    """
    Base class for all embeddings.
    """
    filename = DEFAULT_EMBEDDING_FILE
    config_filename = DEFAULT_EMBEDDING_CONFIG_FILE
    subfolder = DEFAULT_EMBEDDING_SUBFOLDER

    def __init__(self, config: EmbeddingConfig, **kwargs):
        self.config = config.update(kwargs)

    def build(self):
        raise NotImplementedError

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        # TODO
        ...

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        config_filename=None,
        subfolder=None,
        **kwargs,
    ):
        config_filename = config_filename or cls.config_filename
        subfolder = subfolder or cls.subfolder

        config = EmbeddingConfig.load(hub_or_local_path, filename=config_filename, subfolder=subfolder, **kwargs)

        embedding = build_embedding(config.name, config, **kwargs)

        return embedding

    def save(
        self,
        path: Union[str, os.PathLike],
        filename: str = None,
        save_config: bool = True,
        config_filename: str = None,
    ):
        raise NotImplementedError

    def push_to_hub(
        self,
        repo_id,
        commit_message=None,
        subfolder=None,
        filename=None,
        config_filename=None,
        private=False,
    ):
        subfolder = subfolder or self.subfolder
        filename = filename or self.filename
        config_filename = config_filename or self.config_filename

        api = HfApi()
        # create remote repo
        api.create_repo(repo_id, exist_ok=True)
        # create local repo
        cache_path = get_local_cache_path(repo_id, repo_type="model")
        # save tokenizer.json
        embedding_save_path = os.path.join(cache_path, subfolder, filename)
        os.makedirs(os.path.join(cache_path, subfolder), exist_ok=True)

        if commit_message is None:
            commit_message = "Hezar: Upload embedding and config"

        self.save(embedding_save_path, filename, save_config=False)

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
            path_or_fileobj=embedding_save_path,
            repo_type="model",
            path_in_repo=f"{subfolder}/{filename}",
            commit_message=commit_message,
        )

        logger.info(
            f"Uploaded: {self.__class__.__name__}(name={self.config.name})`"
            f" --> "
            f"{os.path.join(repo_id, subfolder, filename)}"
        )



