import os
from dataclasses import dataclass, field
from typing import List, Union

from huggingface_hub import create_repo
from tokenizers import normalizers

from ..builders import build_preprocessor
from ..configs import PreprocessorConfig
from ..constants import DEFAULT_NORMALIZER_CONFIG_FILE, DEFAULT_PREPROCESSOR_SUBFOLDER
from ..registry import register_preprocessor
from ..utils import get_logger
from .preprocessor import Preprocessor


logger = get_logger(__name__)


@dataclass
class NormalizerConfig(PreprocessorConfig):
    name: str = field(default="normalizer")
    replace_pattern: str = None
    nfkd: bool = True
    nfkc: bool = True


@register_preprocessor("normalizer", config_class=NormalizerConfig)
class Normalizer(Preprocessor):
    """
    A simple configurable text normalizer
    """
    preprocessor_subfolder = DEFAULT_PREPROCESSOR_SUBFOLDER
    normalizer_config_file = DEFAULT_NORMALIZER_CONFIG_FILE

    def __init__(self, config: NormalizerConfig, **kwargs):
        super().__init__(config, **kwargs)

    def __call__(
        self,
        inputs: Union[str, List[str]],
        replace_pattern: str = None,
        nfkd: bool = None,
        nfkc: bool = None,
        **kwargs,
    ):
        if isinstance(inputs, str):
            inputs = [inputs]

        nfkd = nfkd or self.config.nfkd
        nfkc = nfkc or self.config.nfkc

        if nfkd:
            inputs = [normalizers.NFKD().normalize_str(x) for x in inputs]
        if nfkc:
            inputs = [normalizers.NFKC().normalize_str(x) for x in inputs]

        # TODO add support for other normalizations

        return inputs

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        subfolder=None,
        config_filename=None,
        **kwargs
    ):
        config_filename = config_filename or cls.normalizer_config_file
        subfolder = subfolder or cls.preprocessor_subfolder
        config = NormalizerConfig.load(
            hub_or_local_path,
            filename=config_filename,
            subfolder=subfolder,
        )
        normalizer = build_preprocessor(config.name, config, **kwargs)
        return normalizer

    def push_to_hub(
        self,
        repo_id,
        commit_message: str = None,
        subfolder: str = None,
        config_filename: str = None,
        private: bool = None,
    ):
        """
        Push normalizer config and other optional files to the Hub.

        Args:
            repo_id: Repo id on the Hub
            commit_message: Commit message
            subfolder: Optional subfolder for the normalizer
            config_filename: Optional normalizer config filename
            private: Whether to create a private repo if it does not exist already
        """
        subfolder = subfolder or self.preprocessor_subfolder
        config_filename = config_filename or self.normalizer_config_file

        # create remote repo
        create_repo(repo_id, exist_ok=True, private=private)

        if commit_message is None:
            commit_message = "Hezar: Upload tokenizer and config"

        # upload config
        self.config.push_to_hub(
            repo_id=repo_id,
            filename=config_filename,
            subfolder=subfolder,
            commit_message=commit_message,
        )

    def save(
        self,
        path,
        config_filename=None,
        **kwargs,
    ):
        config_filename = config_filename or self.normalizer_config_file
        os.makedirs(path, exist_ok=True)
        self.config.save(path, filename=config_filename, subfolder=self.preprocessor_subfolder)
