import os
from typing import List
from dataclasses import dataclass

from transformers import AutoTokenizer, PretrainedConfig

from ..configs import PreprocessorConfig
from ..hub_utils import resolve_hub_path, get_local_cache_path
from ..utils import hezar_config_to_hf_config
from ..registry import build_preprocessor
from ..constants import HEZAR_TMP_DIR
from ..preprocessors import Preprocessor, register_preprocessor


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

    preprocessor_filename = "tokenizer_config.yaml"

    def __init__(self, config: TokenizerConfig, **kwargs):
        super().__init__(config, **kwargs)
        pretrained_path = config.pop("pretrained_path")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path,
            config=PretrainedConfig(**self.config),
            cache_dir=HEZAR_TMP_DIR,
            **self.config,
        )

    def __call__(self, inputs: List[str], **kwargs):
        outputs = self._tokenizer(inputs, **kwargs)
        return outputs

    def push_to_hub(self, hub_path, **kwargs):
        self._tokenizer.push_to_hub(hub_path, **kwargs)

    def save(self, path, **kwargs):
        self._tokenizer.save_pretrained(path, **kwargs)

    @classmethod
    def load(cls, hub_or_local_path, save_to_cache=False, **kwargs):
        hub_or_local_path = resolve_hub_path(hub_or_local_path)
        # Build preprocessor wih config
        preprocessor = build_preprocessor("tokenizer", pretrained_path=hub_or_local_path, **kwargs)
        if save_to_cache:
            cache_path = get_local_cache_path(hub_or_local_path, repo_type="model")
            preprocessor.save(cache_path)
        return preprocessor
