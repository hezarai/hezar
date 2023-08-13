import os
from dataclasses import dataclass, field
from typing import List

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from ...constants import DEFAULT_TOKENIZER_CONFIG_FILE, DEFAULT_TOKENIZER_FILE, HEZAR_CACHE_DIR
from ...registry import register_preprocessor
from .tokenizer import Tokenizer, TokenizerConfig


@dataclass
class SentencePieceBPEConfig(TokenizerConfig):
    name: str = "sentencepiece_bpe_tokenizer"
    max_length: int = 512
    truncation_strategy: str = "longest_first"
    truncation_direction: str = "right"
    stride: int = 0
    padding_strategy: str = "longest"
    padding_direction: str = "right"
    special_tokens: List[str] = field(
        default_factory=lambda: [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
            "<|endoftext|>",
            "<|startoftext|>",
            "<nl>",
            "<hs>",
            "<sep>",
            "<cls>",
        ],
    )
    unk_token: str = "<unk>"
    pad_token_id: int = 0
    pad_token_type_id: int = 0
    pad_token: str = "<pad>"
    pad_to_multiple_of: int = 0
    dropout: float = None
    continuing_subword_prefix: str = ""
    replacement: str = "_"
    add_prefix_space: bool = True
    end_of_word_suffix: str = ""
    fuse_unk: bool = False
    vocab_size: int = 30000
    min_frequency: int = 2
    limit_alphabet: int = 1000
    initial_alphabet: list = field(default_factory=list)
    show_progress: bool = True


@register_preprocessor("sentencepiece_bpe_tokenizer", config_class=SentencePieceBPEConfig)
class SentencePieceBPETokenizer(Tokenizer):
    """
    A standard SentencePiece BPE tokenizer using 🤗HuggingFace Tokenizers

    Args:
        config: Preprocessor config for the tokenizer
        **kwargs: Extra/manual config parameters
    """

    tokenizer_filename = DEFAULT_TOKENIZER_FILE
    tokenizer_config_filename = DEFAULT_TOKENIZER_CONFIG_FILE
    token_ids_name = "token_ids"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def build(self):
        pretrained_path = self.config.get("pretrained_path")
        if pretrained_path:
            if not os.path.isdir(pretrained_path):
                tokenizer_path = hf_hub_download(
                    pretrained_path,
                    filename=self.tokenizer_filename,
                    subfolder=self.preprocessor_subfolder,
                    cache_dir=HEZAR_CACHE_DIR,
                    resume_download=True,
                )

            else:
                tokenizer_path = os.path.join(
                    pretrained_path,
                    self.preprocessor_subfolder,
                    self.tokenizer_filename,
                )
            tokenizer = HFTokenizer.from_file(tokenizer_path)
        else:
            tokenizer = HFTokenizer(
                models.BPE(
                    dropout=self.config.dropout,
                    unk_token=self.config.unk_token,
                    continuing_subword_prefix=self.config.continuing_subword_prefix,
                    end_of_word_suffix=self.config.end_of_word_suffix,
                    fuse_unk=self.config.fuse_unk,
                )
            )
            tokenizer.add_special_tokens(self.config.special_tokens)
            tokenizer.normalizer = normalizers.NFKC()  # noqa
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

        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,  # noqa
            min_frequency=self.config.min_frequency,  # noqa
            show_progress=self.config.show_progress,  # noqa
            special_tokens=self.config.special_tokens,  # noqa
            initial_alphabet=self.config.initial_alphabet,  # noqa
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(self, dataset: List[str], **train_kwargs):
        """Train the model using the given files"""
        self.config.update(train_kwargs)

        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,  # noqa
            min_frequency=self.config.min_frequency,  # noqa
            show_progress=self.config.show_progress,  # noqa
            special_tokens=self.config.special_tokens,  # noqa
            initial_alphabet=self.config.initial_alphabet,  # noqa
        )
        self._tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))
