import os
from dataclasses import dataclass
from typing import List

from huggingface_hub import HfApi
from transformers import AutoTokenizer, PretrainedConfig

from ..configs import PreprocessorConfig
from ..constants import HEZAR_TMP_DIR
from ..hub_utils import resolve_hub_path, get_local_cache_path
from ..preprocessors import Preprocessor, register_preprocessor
from ..registry import build_preprocessor

__all__ = [
    "Tokenizer",
    "TokenizerConfig",
]


@dataclass
class TokenizerConfig(PreprocessorConfig):
    pretrained_path: str = None


@register_preprocessor("tokenizer", config_class=TokenizerConfig)
class Tokenizer(Preprocessor):
    """
    A simple wrapper for HuggingFace Tokenizers

    Args:
        config: Preprocessor config for the tokenizer
        kwargs: Extra/manual config parameters
    """

    def __init__(self, config: TokenizerConfig, **kwargs):
        super().__init__(config, **kwargs)
        pretrained_path = config.pop("pretrained_path")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path,
            config=PretrainedConfig(**self.config),
            cache_dir=HEZAR_TMP_DIR,
            subfolder=self.preprocessor_subfolder,
            **self.config,
        )

    def __call__(self, inputs: List[str], **kwargs):
        outputs = self._tokenizer(inputs, **kwargs)
        return outputs

    def push_to_hub(self, hub_path, **kwargs):
        save_path = f"{get_local_cache_path(hub_path, repo_type='model')}/{self.preprocessor_subfolder}"
        hub_path = resolve_hub_path(hub_path)
        self._tokenizer.save_pretrained(
            save_path,
            repo_id=hub_path,
            **kwargs,
        )
        api = HfApi()
        api.upload_folder(
            repo_id=hub_path,
            folder_path=save_path,
            repo_type="model",
            path_in_repo=self.preprocessor_subfolder,
        )

    def save(self, path, **kwargs):
        save_path = os.path.join(path, self.preprocessor_subfolder)
        self._tokenizer.save_pretrained(save_path, subfolder=self.preprocessor_subfolder, **kwargs)

    @classmethod
    def load(cls, hub_or_local_path, save_to_cache=False, **kwargs):
        hub_or_local_path = resolve_hub_path(hub_or_local_path)
        # Build preprocessor wih config
        preprocessor = build_preprocessor("tokenizer", pretrained_path=hub_or_local_path, **kwargs)
        if save_to_cache:
            cache_path = get_local_cache_path(hub_or_local_path, repo_type="model")
            preprocessor.save(cache_path)
        return preprocessor

    @property
    def tokenizer(self):
        return self._tokenizer
