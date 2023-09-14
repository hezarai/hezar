import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple, Union

from ..builders import build_preprocessor
from ..configs import PreprocessorConfig
from ..constants import DEFAULT_NORMALIZER_CONFIG_FILE, DEFAULT_PREPROCESSOR_SUBFOLDER, Backends
from ..registry import register_preprocessor
from ..utils import Logger, is_backend_available
from .preprocessor import Preprocessor

if is_backend_available(Backends.TOKENIZERS):
    from tokenizers import Regex, normalizers

_required_backends = [
    Backends.TOKENIZERS,
]

logger = Logger(__name__)


@dataclass
class TextNormalizerConfig(PreprocessorConfig):
    name = "text_normalizer"
    replace_patterns: Union[List[Tuple[str, str]], List[List[str]], List[Dict[str, List]]] = None
    nfkd: bool = True
    nfkc: bool = True

    def __post_init__(self):
        # convert a list of dicts in replace_patterns to a list of tuples
        if self.replace_patterns is not None and len(self.replace_patterns):
            if isinstance(self.replace_patterns, Mapping):
                patterns = []
                for v in self.replace_patterns.values():
                    patterns += v
                self.replace_patterns = patterns


@register_preprocessor("text_normalizer", config_class=TextNormalizerConfig)
class TextNormalizer(Preprocessor):
    """
    A simple configurable text normalizer
    """
    required_backends = _required_backends

    preprocessor_subfolder = DEFAULT_PREPROCESSOR_SUBFOLDER
    normalizer_config_file = DEFAULT_NORMALIZER_CONFIG_FILE

    def __init__(self, config: TextNormalizerConfig, **kwargs):
        super().__init__(config, **kwargs)

    def __call__(
        self,
        inputs: Union[str, List[str]],
        replace_patterns: Union[List[Tuple[str, str]], List[List[str]]] = None,
        nfkd: bool = None,
        nfkc: bool = None,
        **kwargs,
    ):
        if isinstance(inputs, str):
            inputs = [inputs]

        nfkd = nfkd or self.config.nfkd
        nfkc = nfkc or self.config.nfkc
        replace_patterns = replace_patterns or self.config.replace_patterns

        if nfkd:
            inputs = [normalizers.NFKD().normalize_str(x) for x in inputs]
        if nfkc:
            inputs = [normalizers.NFKC().normalize_str(x) for x in inputs]

        if replace_patterns is not None:
            replacer = normalizers.Sequence(
                [normalizers.Replace(Regex(src), trg) for src, trg in self.config.replace_patterns]  # noqa
            )
            inputs = [replacer.normalize_str(x) for x in inputs]

        return inputs

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        subfolder=None,
        config_filename=None,
        **kwargs
    ) -> "TextNormalizer":
        config_filename = config_filename or cls.normalizer_config_file
        subfolder = subfolder or cls.preprocessor_subfolder
        config = TextNormalizerConfig.load(
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

        if commit_message is None:
            commit_message = "Hezar: Upload normalizer"

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
        subfolder=None,
        config_filename=None,
    ):
        config_filename = config_filename or self.normalizer_config_file
        subfolder = subfolder or self.preprocessor_subfolder
        os.makedirs(path, exist_ok=True)
        self.config.save(path, filename=config_filename, subfolder=subfolder)
