from dataclasses import dataclass
from typing import Union, List

from tokenizers import Regex, normalizers

from .preprocessor import Preprocessor
from ..configs import PreprocessorConfig
from ..registry import register_preprocessor


@dataclass
class NormalizerConfig(PreprocessorConfig):
    name = "normalizer"
    replace_pattern: str = None
    nfkd: bool = True
    nfkc: bool = True


@register_preprocessor("normalizer", config_class=NormalizerConfig)
class Normalizer(Preprocessor):
    def __init__(self, config: NormalizerConfig, **kwargs):
        super().__init__(config, **kwargs)

    def __call__(
        self,
        inputs: Union[str, List[str]],
        replace_pattern: str = None,
        **kwargs,
    ):
        # TODO
        return inputs
