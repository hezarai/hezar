import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Union

import torch
from huggingface_hub import create_repo, upload_file
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import Model

from ...builders import build_preprocessor
from ...configs import PreprocessorConfig
from ...constants import DEFAULT_TOKENIZER_CONFIG_FILE, DEFAULT_TOKENIZER_FILE
from ...data.utils import convert_batch_dict_dtype
from ...utils import get_logger
from ..preprocessor import Preprocessor


logger = get_logger(__name__)


@dataclass
class TokenizerConfig(PreprocessorConfig):
    name: str = "tokenizer"
    pretrained_path: str = None
    max_length: int = None
    truncation_strategy: str = None
    truncation_direction: str = None
    stride: int = None
    padding_strategy: str = None
    padding_direction: str = None
    pad_to_multiple_of: int = None
    pad_token_id: int = None
    pad_token: str = None
    pad_token_type_id: int = None
    unk_token: str = None


class Tokenizer(Preprocessor):
    """
    Base tokenizer class. Mostly copied from :class: ~tokenizers.implementations.BaseTokenizer

    Args:
        config: A TokenizerConfig instance
        **kwargs: Extra config parameters that merge into the main config
    """

    tokenizer_filename = DEFAULT_TOKENIZER_FILE
    tokenizer_config_filename = DEFAULT_TOKENIZER_CONFIG_FILE
    token_ids_name = "token_ids"

    def __init__(self, config: TokenizerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._tokenizer = self.build()

    def build(self) -> HFTokenizer:
        raise NotImplementedError

    def encode(self, inputs, is_pretokenized: bool = False, add_special_tokens: bool = True):
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, list) and is_pretokenized:
            if isinstance(inputs[0], str):
                inputs = [inputs]
        return self._tokenizer.encode_batch(inputs, is_pretokenized, add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if isinstance(ids[0], list):
            return self._tokenizer.decode_batch(ids, skip_special_tokens=skip_special_tokens)
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def pad_encoded_batch(
        self,
        inputs,
        padding: Literal["longest", "max_length"] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        skip_keys: list = None,
    ):
        """
        Given a batch of encoded inputs, add padding to all of them so that the inputs are of the same length.

        Args:
            inputs: Input batch of encoded tokens
            padding: Padding type, could be one of ["longest", "max_length"]
            max_length: Max input length (only if padding is set to "max_length")
            return_tensors: The type of tensors to return
            skip_keys: A list of keys to skip padding

        Returns:

        """

        if isinstance(inputs, (list, tuple)) and isinstance(inputs[0], Mapping):
            inputs = {key: [example[key] for example in inputs] for key in inputs[0].keys()}

        inputs_max_length = max([len(x) for x in inputs[self.token_ids_name]])
        # resolve padding and max_length parameters
        padding = padding or self.config.padding_strategy

        if padding is None:
            if max_length is not None:
                padding = "max_length"
            elif max_length is None:
                logger.warning("Both padding and max_length are None so the inputs cannot be padded!")
                return inputs

        if padding == "longest":
            if max_length is not None:
                logger.warning("Setting padding='longest' and max_length is not valid. You must set one of them"
                               "and leave the other as None. Falling back to padding='longest'")

            inputs_length = inputs_max_length

        elif padding == "max_length":
            if max_length is None:
                logger.warning("Setting padding='max_length' but no max_length value is provided in the function "
                               "parameters nor the tokenizer config! Falling back to padding='longest'")
                inputs_length = inputs_max_length
            else:
                # TODO implement truncation if possible and remove this condition
                if max_length <= inputs_max_length:
                    logger.warning(f"Cannot set max_length to {max_length} "
                                   f"while max input length is {inputs_max_length}!"
                                   f"Falling back to padding='longest' "
                                   f"since truncation is not available yet in Hezar :(")
                    inputs_length = inputs_max_length
                else:
                    inputs_length = max_length

        inputs = convert_batch_dict_dtype(inputs, dtype="list", skip_keys=skip_keys)

        skip_keys = skip_keys or []
        for key, batch in inputs.items():
            if key in skip_keys:
                continue
            padding_id = 0 if key == "attention_mask" else self.config.pad_token_id
            padded_batch = []
            for x in batch:
                difference = inputs_length - len(x)
                paddings = [padding_id] * difference
                padded_x = x + paddings if self.config.padding_direction == "right" else paddings + x
                padded_batch.append(padded_x)
            inputs[key] = padded_batch

        inputs = convert_batch_dict_dtype(inputs, dtype=return_tensors)

        return inputs

    def __call__(
        self,
        inputs: List[str],
        device: Union[str, torch.device] = None,
        add_special_tokens: bool = True,
        padding_strategy=None,
        truncation_strategy=None,
        max_length: int = None,
        return_tensors: str = "list",
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int = None,
        return_tokens: bool = None,
        return_token_type_ids: bool = None,
        return_attention_mask: bool = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = True,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_word_ids: bool = False,
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
            **kwargs: Extra arguments reside here and therefore ignored

        Returns:
            A dictionary of encoded inputs like
                {"token_ids": [batch_size x input_len], "attention_mask": [batch_size x input_len], ...}
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        padding_strategy = padding_strategy or self.config.padding_strategy
        truncation_strategy = truncation_strategy or self.config.truncation_strategy
        max_length = max_length or self.config.max_length
        pad_to_multiple_of = pad_to_multiple_of or self.config.pad_to_multiple_of

        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            padding_side=self.config.padding_direction,
            truncation_side=self.config.truncation_direction,
            max_length=max_length,
            stride=self.config.stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        encodings = self.encode(
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
                return_word_ids=return_word_ids,
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

        if return_tensors == "list" or return_tensors is None:
            sanitized_outputs = {
                key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                for key, value in sanitized_outputs.items()
            }

        outputs = convert_batch_dict_dtype(sanitized_outputs, dtype=return_tensors)
        if device and return_tensors == "pt":
            outputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}

        return outputs

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
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == "no_truncation":
            if self.truncation is not None:
                self.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy,
                "direction": truncation_side,
            }
            if self.truncation is None:
                current = None
            else:
                current = {k: self.truncation.get(k, None) for k in target}

            if current != target:
                self.enable_truncation(**target)

            if padding_strategy == "no_padding":
                if self.padding is not None:
                    self.no_padding()
            else:
                length = max_length if self.config.padding_strategy == "max_length" else None
                target = {
                    "length": length,
                    "direction": padding_side,
                    "pad_id": self.config.pad_token_id,
                    "pad_token": self.config.pad_token,
                    "pad_type_id": self.config.pad_token_type_id,
                    "pad_to_multiple_of": pad_to_multiple_of,
                }
                if self.padding != target:
                    self.enable_padding(**target)

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
        return_word_ids: bool = False,
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
            if return_word_ids:
                encoding_dict["word_ids"].append(e.word_ids)

        return encoding_dict

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        return self._tokenizer.num_special_tokens_to_add(is_pair)

    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=with_added_tokens)

    def enable_padding(
        self,
        direction: str = "right",
        pad_to_multiple_of: int = None,
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: int = None,
    ):
        return self._tokenizer.enable_padding(
            direction=direction,
            pad_to_multiple_of=pad_to_multiple_of,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            length=length,
        )

    def no_padding(self):
        return self._tokenizer.no_padding()

    def enable_truncation(self, max_length, stride=0, strategy="longest_first", direction="right"):
        return self._tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy, direction=direction)

    def no_truncation(self):
        return self._tokenizer.no_truncation()

    def add_tokens(self, tokens) -> int:
        return self._tokenizer.add_tokens(tokens)

    def add_special_tokens(self, special_tokens) -> int:
        return self._tokenizer.add_special_tokens(special_tokens)

    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        return self._tokenizer.id_to_token(id)

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        config_filename=None,
        subfolder=None,
        **kwargs,
    ):
        config_filename = config_filename or cls.tokenizer_config_filename
        subfolder = subfolder or cls.preprocessor_subfolder
        config = TokenizerConfig.load(
            hub_or_local_path,
            filename=config_filename,
            subfolder=subfolder,
        )
        config.pretrained_path = hub_or_local_path
        tokenizer = build_preprocessor(config.name, config, **kwargs)
        return tokenizer

    def save(self, path, save_config=True, pretty=True):
        os.makedirs(path, exist_ok=True)
        # save config
        if save_config:
            self.config.save(path, filename=self.tokenizer_config_filename, subfolder=self.preprocessor_subfolder)
        # save tokenizer.json
        save_path = os.path.join(path, self.preprocessor_subfolder, self.tokenizer_filename)
        self._tokenizer.save(save_path, pretty=pretty)

    def push_to_hub(
        self,
        repo_id,
        commit_message=None,
        subfolder=None,
        tokenizer_filename=None,
        config_filename=None,
        private=False,
    ):
        """
        Push tokenizer and config to the Hub

        Args:
            repo_id: The path (id or repo name) on the hub
            commit_message: Commit message for this push
            subfolder: subfolder to save the files
            tokenizer_filename: tokenizer filename
            config_filename: tokenizer config filename
            private: If the repo should be private (ignored if the repo exists)
        """
        subfolder = subfolder or self.preprocessor_subfolder
        tokenizer_filename = tokenizer_filename or self.tokenizer_filename
        config_filename = config_filename or self.tokenizer_config_filename

        # create remote repo
        create_repo(repo_id, exist_ok=True, private=private)
        # save to tmp and prepare for push
        cache_path = tempfile.mkdtemp()
        # save tokenizer.json
        tokenizer_save_path = os.path.join(cache_path, subfolder, tokenizer_filename)
        os.makedirs(os.path.join(cache_path, subfolder), exist_ok=True)
        self._tokenizer.save(tokenizer_save_path, pretty=True)

        if commit_message is None:
            commit_message = "Hezar: Upload tokenizer and config"

        # upload config
        self.config.push_to_hub(
            repo_id=repo_id,
            filename=config_filename,
            subfolder=subfolder,
            commit_message=commit_message,
        )
        # upload tokenizer
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=tokenizer_save_path,
            repo_type="model",
            path_in_repo=f"{subfolder}/{tokenizer_filename}",
            commit_message=commit_message,
        )
        logger.info(
            f"Uploaded: {self.__class__.__name__}(name={self.config.name})`"
            f" --> "
            f"{os.path.join(repo_id, subfolder, tokenizer_filename)}"
        )

    @property
    def model(self) -> Model:
        return self._tokenizer.model

    @model.setter
    def model(self, model: Model):
        self._tokenizer.model = model  # noqa

    @property
    def decoder(self) -> Decoder:
        return self._tokenizer.decoder

    @decoder.setter
    def decoder(self, decoder: Decoder):
        self._tokenizer.decoder = decoder  # noqa

    @property
    def padding(self):
        return self._tokenizer.padding

    @property
    def truncation(self) -> dict:
        return self._tokenizer.truncation
