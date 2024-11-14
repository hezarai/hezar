from dataclasses import dataclass, field
from typing import List

from ...constants import DEFAULT_TOKENIZER_CONFIG_FILE, DEFAULT_TOKENIZER_FILE, Backends
from ...registry import register_preprocessor
from ...utils import is_backend_available
from .tokenizer import Tokenizer, TokenizerConfig


if is_backend_available(Backends.TOKENIZERS):
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, pre_tokenizers, trainers

_required_backends = [
    Backends.TOKENIZERS,
]


@dataclass
class SentencePieceUnigramConfig(TokenizerConfig):
    name = "sentencepiece_unigram_tokenizer"
    truncation_side: str = "right"
    stride: int = 0
    padding_side: str = "right"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    sep_token: str = "<sep>"
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    mask_token: str = "<mask>"
    additional_special_tokens: List[str] = None
    pad_to_multiple_of: int = 0
    dropout: float = None
    continuing_subword_prefix: str = ""
    replacement: str = "_"
    add_prefix_space: bool = True
    end_of_word_suffix: str = ""
    fuse_unk: bool = False
    vocab_size: int = 8000
    min_frequency: int = 2
    limit_alphabet: int = 1000
    initial_alphabet: list = field(default_factory=list)
    show_progress: bool = True


@register_preprocessor("sentencepiece_unigram_tokenizer", config_class=SentencePieceUnigramConfig)
class SentencePieceUnigramTokenizer(Tokenizer):
    """
    A standard SentencePiece Unigram tokenizer using 🤗HuggingFace Tokenizers

    Args:
        config: Preprocessor config for the tokenizer
        **kwargs: Extra/manual config parameters
    """

    required_backends = _required_backends

    tokenizer_filename = DEFAULT_TOKENIZER_FILE
    tokenizer_config_filename = DEFAULT_TOKENIZER_CONFIG_FILE
    token_ids_name = "token_ids"

    def __init__(self, config, tokenizer_file=None, **kwargs):
        super().__init__(config, tokenizer_file=tokenizer_file, **kwargs)

    def build(self):
        tokenizer = HFTokenizer(models.Unigram())  # noqa
        tokenizer.normalizer = normalizers.Sequence(  # noqa
            [  # noqa
                normalizers.Nmt(),  # noqa
                normalizers.NFKC(),  # noqa
                normalizers.Replace(Regex(" {2,}"), " "),  # noqa
            ]  # noqa
        )  # noqa
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(  # noqa
            replacement=self.config.replacement, add_prefix_space=self.config.add_prefix_space
        )
        tokenizer.decoder = decoders.Metaspace(  # noqa
            replacement=self.config.replacement, add_prefix_space=self.config.add_prefix_space
        )

        return tokenizer

    def train(self, files: List[str], **train_kwargs):
        """Train the model using the given files"""
        self.config.update(train_kwargs)

        trainer = trainers.UnigramTrainer(
            vocab_size=self.config.vocab_size,  # noqa
            special_tokens=self.config.special_tokens,  # noqa
            show_progress=self.config.show_progress,  # noqa
            initial_alphabet=self.config.initial_alphabet,  # noqa
            unk_token=self.config.unk_token,  # noqa
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(self, dataset: List[str], **train_kwargs):
        """Train the model using the given files"""
        self.config.update(train_kwargs)

        trainer = trainers.UnigramTrainer(
            vocab_size=self.config.vocab_size,  # noqa
            special_tokens=self.config.special_tokens,  # noqa
            show_progress=self.config.show_progress,  # noqa
            initial_alphabet=self.config.initial_alphabet,  # noqa
            unk_token=self.config.unk_token,  # noqa
        )
        self._tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))
