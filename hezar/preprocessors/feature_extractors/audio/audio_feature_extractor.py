from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ....builders import build_preprocessor
from ....configs import PreprocessorConfig
from ....constants import DEFAULT_FEATURE_EXTRACTOR_CONFIG_FILE
from ....utils import convert_batch_dict_dtype
from ...preprocessor import Preprocessor


@dataclass
class AudioFeatureExtractorConfig(PreprocessorConfig):
    feature_size: int = None
    sampling_rate: int = 16000
    padding: str = None
    padding_value: float = 0.0
    padding_side: str = None


class AudioFeatureExtractor(Preprocessor):
    """
    Base class for all audio feature extractors.
    """
    model_input_name = "input_features"
    config_filename = DEFAULT_FEATURE_EXTRACTOR_CONFIG_FILE

    def __init__(self, config: AudioFeatureExtractorConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def __call__(self, inputs, **kwargs):
        raise NotImplementedError

    def pad(
        self,
        processed_features,
        padding=None,
        max_length=None,
        truncation=None,
        pad_to_multiple_of=None,
        return_attention_mask=None,
        return_tensors=None,
    ):
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Args:
            processed_features: Processed inputs to add padding to
            padding: Padding strategy which can be `longest`, `max_length`, `False`
            max_length: Max input length (Only effective if padding is `max_length` too, ignored otherwise)
            truncation: Whether to truncate long inputs or not
            pad_to_multiple_of: If set will pad the sequence to a multiple of the provided value.
            return_attention_mask:  Whether to return the attention mask.
            return_tensors: Tensors return type among `pt`, `np`, `list`
        """
        return_attention_mask = return_attention_mask or self.config.return_attention_mask
        padding = padding or self.config.padding
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], Mapping):
            processed_features = {
                key: np.array([example[key] for example in processed_features]) for key in processed_features[0].keys()
            }
        if self.model_input_name not in processed_features:
            raise ValueError(f"Processed inputs must have a `{self.model_input_name}` key!\n"
                             f"Provided keys: {list(processed_features.keys())}")

        required_input = processed_features[self.model_input_name]

        if len(required_input) == 0:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non-empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]

        # processed_features = convert_batch_dict_dtype(processed_features, dtype="np")
        padding_strategy = self._get_padding_strategy(padding=padding, max_length=max_length)

        required_input = processed_features[self.model_input_name]

        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError("Some items in the output dictionary have a different batch size than others.")

        truncated_inputs = []
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in processed_features.items()}
            # truncation
            inputs_slice = self._truncate(
                inputs,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)

        if padding_strategy == "longest":
            # make sure that `max_length` cannot be longer than the longest truncated length
            max_length = max(len(input_slice[self.model_input_name]) for input_slice in truncated_inputs)
            padding_strategy = "max_length"

        batch_outputs = {}
        for i in range(batch_size):
            # padding
            outputs = self._pad(
                truncated_inputs[i],
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if value.dtype is np.dtype(np.float64):
                    value = value.astype(np.float32)
                batch_outputs[key].append(value)

        batch_outputs = {k: np.array(v) for k, v in batch_outputs.items()}

        padded_features = convert_batch_dict_dtype(batch_outputs, dtype=return_tensors)

        return padded_features

    def _pad(
        self,
        processed_features,
        max_length=None,
        padding_strategy=None,
        pad_to_multiple_of=None,
        return_attention_mask=None,
    ) -> dict:
        """
        Pad inputs based on padding strategy and max length

        Args:
            processed_features: Processed inputs to add padding to
            padding_strategy: Padding strategy which can be `longest`, `max_length`, `False`
            max_length: Max input length (Only effective if padding is `max_length` too, ignored otherwise)
            pad_to_multiple_of: If set will pad the sequence to a multiple of the provided value.
            return_attention_mask:  Whether to return the attention mask.

        Returns:
            Batch dict of the padded features
        """
        required_input = processed_features[self.model_input_name]

        if padding_strategy == "longest":
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy is not None and len(required_input) < max_length

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = np.ones(len(required_input), dtype=np.int32)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.config.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = ((0, difference), (0, 0)) if self.config.feature_size > 1 else (0, difference)
                processed_features[self.model_input_name] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.config.padding_value
                )
            elif self.config.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = ((difference, 0), (0, 0)) if self.config.feature_size > 1 else (difference, 0)
                processed_features[self.model_input_name] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.config.padding_value
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.config.padding_side))

        return processed_features

    def _get_padding_strategy(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """
        if padding == "longest" or padding is True:
            padding_strategy = "longest"
            if self.config.padding_value is None:
                raise ValueError(f"Setting padding to `{padding_strategy}`, but `config.padding_value` is `None`!")

        elif padding == "max_length":
            if max_length is None:
                raise ValueError("Setting `padding=max_length` but leaving `max_length` as `None` is invalid!")
            padding_strategy = "max_length"

        else:
            padding_strategy = None

        return padding_strategy

    def _truncate(
        self,
        processed_features,
        max_length: int = None,
        pad_to_multiple_of: int = None,
        truncation: bool = None,
    ):
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features: Dictionary of input values
            max_length: maximum length of the returned list and optionally padding length
            pad_to_multiple_of: Integer if set will pad the sequence to a multiple of the provided value.
            truncation: Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        if not truncation:
            return processed_features
        elif truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        required_input = processed_features[self.model_input_name]

        # find `max_length` that fits `pad_to_multiple_of`
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length

        if needs_to_be_truncated:
            processed_features[self.model_input_name] = processed_features[self.model_input_name][:max_length]
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]

        return processed_features

    def save(
        self,
        path,
        subfolder=None,
        config_filename=None,
    ):
        subfolder = subfolder or self.preprocessor_subfolder
        config_filename = config_filename or self.config_filename

        self.config.save(path, filename=config_filename, subfolder=subfolder)

    def push_to_hub(
        self,
        repo_id,
        subfolder=None,
        commit_message=None,
        private=None,
        config_filename=None,
    ):
        subfolder = subfolder or self.preprocessor_subfolder
        config_filename = config_filename or self.config_filename

        if commit_message is None:
            commit_message = "Hezar: Upload feature extractor"

        self.config.push_to_hub(
            repo_id,
            subfolder=subfolder,
            filename=config_filename,
            private=private,
            commit_message=commit_message,
        )

    @classmethod
    def load(
        cls,
        hub_or_local_path,
        subfolder: str = None,
        config_filename: str = None,
        force_return_dict: bool = False,
        **kwargs
    ):
        subfolder = subfolder or cls.preprocessor_subfolder
        config_filename = config_filename or cls.config_filename

        config = AudioFeatureExtractorConfig.load(
            hub_or_local_path,
            subfolder=subfolder,
            filename=config_filename,
        )

        feature_extractor = build_preprocessor(config.name, config=config, **kwargs)

        return feature_extractor


