from dataclasses import dataclass
from typing import List, Union

import numpy as np

from ....registry import register_preprocessor
from ....utils import Logger, convert_batch_dict_dtype, mel_filter_bank, spectrogram, window_function
from .audio_feature_extractor import AudioFeatureExtractor, AudioFeatureExtractorConfig


logger = Logger(__name__)


@dataclass
class WhisperFeatureExtractorConfig(AudioFeatureExtractorConfig):
    name = "whisper_feature_extractor"
    feature_size: int = 80
    sampling_rate: int = 16000
    hop_length: int = 160
    chunk_length: int = 30
    n_fft: int = 400
    padding: str = "longest"
    padding_value: float = 0.0
    padding_side: str = "right"
    return_attention_mask: bool = False


@register_preprocessor("whisper_feature_extractor", config_class=WhisperFeatureExtractorConfig)
class WhisperFeatureExtractor(AudioFeatureExtractor):
    """
    A feature extractor for Whisper model.

    This feature extractor inherits from `AudioFeatureExtractor` which contains most of the main methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.
    """
    def __init__(self, config: WhisperFeatureExtractorConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.n_samples = self.config.chunk_length * self.config.sampling_rate
        self.nb_max_frames = self.n_samples // self.config.hop_length
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.config.n_fft // 2,
            num_mel_filters=self.config.feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self.config.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: int = None,
        return_tensors: str = None,
        return_attention_mask: bool = None,
        padding: str = "max_length",
        max_length: int = None,
        sampling_rate: int = None,
        do_normalize: bool = None,
        **kwargs,
    ):
        sampling_rate = sampling_rate or self.config.sampling_rate
        if sampling_rate is not None:
            if sampling_rate != self.config.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.config.sampling_rate}. Please make sure that the provided `raw_speech` "
                    f"input was sampled with {self.config.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = {"input_features": raw_speech}

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask or do_normalize,
            return_tensors="np",
        )

        # zero-mean and unit-variance normalization
        if do_normalize:
            padded_inputs["input_features"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_features"],
                attention_mask=padded_inputs["attention_mask"],
                padding_value=self.config.padding_value,
            )
            padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)

        # make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        input_features = [self._np_extract_fbank_features(waveform) for waveform in input_features[0]]

        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        if return_attention_mask:
            # rescale from sample (48000) to feature (3000)
            padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: self.config.hop_length]

        if return_tensors is not None:
            padded_inputs = {k: np.asarray(v) for k, v in padded_inputs.items()}
            padded_inputs = convert_batch_dict_dtype(padded_inputs, dtype=return_tensors)

        return padded_inputs

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        log_spec = spectrogram(
            waveform,
            window_function(self.config.n_fft, "hann"),
            frame_length=self.config.n_fft,
            hop_length=self.config.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )
        log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values
