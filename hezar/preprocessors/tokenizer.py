"""
Hezar Tokenizer is simply a wrapper for HuggingFace Tokenizers in a way that it can be built, loaded, trained, called in
a single module. As of now a lot of the parts in this file are borrowed from HuggingFace Transformers wrappers for
Tokenizers like ~transformers.PretrainedTokenizerBase.
"""
import os
from dataclasses import dataclass
from typing import List
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordPiece, BPE, Unigram, WordLevel
from tokenizers.decoders import WordPiece as WordPieceDecoder, BPEDecoder
from tokenizers.trainers import WordPieceTrainer, WordLevelTrainer, BpeTrainer, UnigramTrainer

from ..configs import Config, PreprocessorConfig
from ..constants import DEFAULT_TOKENIZER_FILE, DEFAULT_TOKENIZER_CONFIG_FILE
from ..registry import register_preprocessor
from ..utils import resolve_pretrained_path, get_local_cache_path, get_logger
from ..data.utils import convert_batch_dict_dtype
from ..preprocessors import Preprocessor
from ..builders import build_preprocessor

logger = get_logger(__name__)

TOKENIZERS_MAP = {
    "wordpiece": {"encoder": WordPiece, "decoder": WordPieceDecoder, "trainer": WordPieceTrainer},
    "bpe": {"encoder": BPE, "decoder": BPEDecoder, "trainer": BpeTrainer},
    "wordlevel": {"encoder": WordLevel, "trainer": WordLevelTrainer},
    "unigram": {"encoder": Unigram, "trainer": UnigramTrainer},
}


@dataclass
class TokenizerConfig(PreprocessorConfig):
    name = "tokenizer"
    model: str = None
    model_kwargs: dict = None
    pretrained_path: str = None
    max_length: int = 512
    truncation_strategy: str = "no_truncation"
    truncation_direction: str = None
    stride: int = 0
    padding_strategy: str = "no_padding"
    padding_direction: str = None
    pad_to_multiple_of: int = None
    pad_token_id: int = 0
    pad_token: str = "[PAD]"
    pad_token_type_id: int = 0


@dataclass
class TokenizerTrainerConfig(Config):
    model: str = None
    vocab_size: int = None
    special_tokens: list = None
    show_progress: bool = True


@register_preprocessor("tokenizer", config_class=TokenizerConfig)
class Tokenizer(Preprocessor):
    """
    A simple wrapper for HuggingFace `~tokenizers.Tokenizer`

    Args:
        config: Preprocessor config for the tokenizer
        kwargs: Extra/manual config parameters
    """

    tokenizer_filename = DEFAULT_TOKENIZER_FILE
    tokenizer_config_filename = DEFAULT_TOKENIZER_CONFIG_FILE
    token_ids_name = "token_ids"

    def __init__(self, config: TokenizerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._tokenizer = self._build(self.config)

    def _build(self, config: TokenizerConfig):
        pretrained_path = config.pop("pretrained_path")
        if pretrained_path:
            if not os.path.isdir(pretrained_path):
                path = resolve_pretrained_path(pretrained_path)
                tokenizer_path = hf_hub_download(
                    path,
                    filename=self.tokenizer_filename,
                    subfolder=self.preprocessor_subfolder,
                )

            else:
                tokenizer_path = os.path.join(
                    pretrained_path,
                    self.preprocessor_subfolder,
                    self.tokenizer_filename,
                )
            tokenizer = HFTokenizer.from_file(tokenizer_path)
        else:
            if config.model:
                logger.info(
                    f"Creating tokenizer `{config.model}` with default values since no `pretrained_path`"
                    f" was given in config!"
                )
                config.model_kwargs = config.model_kwargs or {}
                encoder = TOKENIZERS_MAP[config.model]["encoder"](**config.model_kwargs)
                tokenizer = HFTokenizer(encoder)
                if "decoder" in TOKENIZERS_MAP[config.model]:
                    tokenizer.decoder = TOKENIZERS_MAP[config.model]["decoder"]()
            else:
                raise ValueError(f"Could not create the tokenizer, because `model` is missing!")
        return tokenizer

    def __call__(
        self,
        inputs: List[str],
        add_special_tokens: bool = True,
        padding_strategy="longest",
        truncation_strategy=None,
        max_length: int = None,
        return_tensors: str = "list",
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int = None,
        return_tokens: bool = None,
        return_token_type_ids: bool = None,
        return_attention_mask: bool = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Tokenize a batch of string inputs and return the relevant properties e.g, token ids, attention mask, etc.

        Args:
            inputs: A list of string inputs to tokenize
            add_special_tokens: Whether to add special tokens or not
            padding_strategy: Determines how to pad inputs
            truncation_strategy: Determines how to truncate inputs
            max_length: Max input length of the sequences
            return_tensors: The type of the returning tensors in the batch e.g, pt, np, list
            stride: Stride level
            is_split_into_words: Are inputs pre-tokenized or raw string inputs
            pad_to_multiple_of: Pad inputs by a factor of this value
            return_tokens: Whether to return tokens lists
            return_token_type_ids: Whether to return token type ids
            return_attention_mask: Whether to return attention masks
            return_overflowing_tokens: Whether to return overflowing tokens
            return_special_tokens_mask: Whether to return special tokens mask
            return_offsets_mapping: Whether to return offsets
            return_length: Whether to return input lengths
            kwargs: Extra arguments reside here and therefore ignored

        Returns:
            A dictionary of encoded inputs like
                {"token_ids": [batch_size x input_len], "attention_mask": [batch_size x input_len], ...}
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        self.set_truncation_and_padding(
            padding_strategy=self.config.padding_strategy,
            truncation_strategy=self.config.truncation_strategy,
            padding_side=self.config.padding_direction,
            truncation_side=self.config.truncation_direction,
            max_length=self.config.max_length,
            stride=self.config.stride,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        encodings = self._tokenizer.encode_batch(
            inputs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )
        encodings_dict = [
            self._convert_encodings(
                encoding=encoding,
                return_tokens=return_tokens,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
            )
            for encoding in encodings
        ]
        # Permute output dict from [batch_0: Dict[key, value], ...] to Dict[key, [batch_0, batch_1, ...], ...]
        sanitized_outputs = {}
        for key in encodings_dict[0].keys():
            stack = [e for item in encodings_dict for e in item[key]]
            sanitized_outputs[key] = stack

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, encodings_ in enumerate(encodings_dict):
                overflow_to_sample_mapping += [i] * len(encodings_["input_ids"])
            sanitized_outputs["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        outputs = convert_batch_dict_dtype(sanitized_outputs, dtype=return_tensors)
        return outputs

    def _convert_encodings(
        self,
        encoding,
        return_tokens: bool = None,
        return_token_type_ids: bool = None,
        return_attention_mask: bool = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
    ):
        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict[self.token_ids_name].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))
            if return_tokens:
                encoding_dict["tokens"].append(e.tokens)

        return encoding_dict

    def set_truncation_and_padding(
        self,
        padding_strategy=None,
        truncation_strategy=None,
        padding_side=None,
        truncation_side=None,
        max_length: int = None,
        stride: int = None,
        pad_to_multiple_of: int = None,
    ):
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == "no_truncation":
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy,
                "direction": truncation_side,
            }
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            if current != target:
                self._tokenizer.enable_truncation(**target)

            if padding_strategy == "no_padding":
                if _padding is not None:
                    self._tokenizer.no_padding()
            else:
                length = max_length if padding_strategy == "max_length" else None
                target = {
                    "length": length,
                    "direction": padding_side,
                    "pad_id": self.config.pad_token_id,
                    "pad_token": self.config.pad_token,
                    "pad_type_id": self.config.pad_token_type_id,
                    "pad_to_multiple_of": pad_to_multiple_of,
                }
                if _padding != target:
                    self._tokenizer.enable_padding(**target)

    def decode(self, token_ids: List[List[int]], **kwargs):
        if not isinstance(token_ids[0], list):
            raise ValueError(f"`token_ids` must be a batch of token ids")
        return self.tokenizer.decode_batch(token_ids, **kwargs)

    def push_to_hub(self, hub_path, **kwargs):
        hub_path = resolve_pretrained_path(hub_path)
        save_dir = get_local_cache_path(hub_path, repo_type="model")
        self.save(save_dir)
        api = HfApi()
        api.upload_folder(
            repo_id=hub_path,
            folder_path=os.path.join(save_dir, self.preprocessor_subfolder),
            repo_type="model",
            path_in_repo=self.preprocessor_subfolder,
        )

    def save(self, path, **kwargs):
        save_path = os.path.join(path, self.preprocessor_subfolder, self.tokenizer_filename)
        self._tokenizer.save(save_path, **kwargs)

    @classmethod
    def load(cls, hub_or_local_path, save_to_cache=False, **kwargs):
        hub_or_local_path = resolve_pretrained_path(hub_or_local_path)
        config = TokenizerConfig.load(
            hub_or_local_path,
            filename=cls.tokenizer_config_filename,
            subfolder=cls.preprocessor_subfolder,
        )
        config.pretrained_path = hub_or_local_path
        # Build preprocessor wih config
        preprocessor = build_preprocessor("tokenizer", config=config, **kwargs)
        if save_to_cache:
            cache_path = get_local_cache_path(hub_or_local_path, repo_type="model")
            preprocessor.save(cache_path)
        return preprocessor

    def train_from_iterator(self, iterator, trainer=None, length=None):
        self.tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)

    @classmethod
    def train(cls, dataset: List[str], config: TokenizerTrainerConfig, tokenizer_config: TokenizerConfig = None):
        tokenizer_model = config.pop("model")
        # remove irrelevant params from config
        config.pop("config_type")
        config.pop("name")
        # build tokenizer and train it
        tokenizer: Tokenizer = build_preprocessor("tokenizer", model=tokenizer_model, config=tokenizer_config)
        trainer = TOKENIZERS_MAP[tokenizer_model]["trainer"](**config)
        logger.info(f"Starting training process...")
        tokenizer.train_from_iterator(dataset, trainer, length=len(dataset))
        logger.info(f"Training finished!")
        return tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer
