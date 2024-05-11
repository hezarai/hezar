from __future__ import annotations

from dataclasses import dataclass

from datasets import Audio, load_dataset

from ...configs import DatasetConfig
from ...constants import Backends, PaddingType, TaskType
from ...preprocessors import AudioFeatureExtractor, Tokenizer
from ...registry import register_dataset
from ..data_collators import SpeechRecognitionDataCollator
from .dataset import Dataset


_required_backends = [Backends.LIBROSA, Backends.DATASETS]


@dataclass
class SpeechRecognitionDatasetConfig(DatasetConfig):
    name = "speech_recognition"
    task = TaskType.SPEECH_RECOGNITION
    path: str = None
    feature_extractor_path: str = None
    tokenizer_path: str = None
    sampling_rate: int = 16000
    audio_array_padding_type: bool | str | PaddingType = "longest"
    max_audio_array_length: int = None
    labels_padding_type: bool | str | PaddingType = "longest"
    labels_max_length: int = None
    audio_file_path_column: str = "path"
    audio_column: str = "audio"
    audio_array_column: str = "array"
    transcript_column: str = "sentence"


@register_dataset("speech_recognition", config_class=SpeechRecognitionDatasetConfig)
class SpeechRecognitionDataset(Dataset):
    required_backends = _required_backends

    def __init__(self, config: SpeechRecognitionDatasetConfig, split=None, **kwargs):
        super().__init__(config, split, **kwargs)
        self.data = self._load(split)
        self.feature_extractor = AudioFeatureExtractor.load(self.config.feature_extractor_path)
        self.tokenizer = Tokenizer.load(self.config.tokenizer_path)
        self.data_collator = SpeechRecognitionDataCollator(
            self.feature_extractor,
            self.tokenizer,
            inputs_padding_type=self.config.audio_array_padding_type,
            inputs_max_length=self.config.max_audio_array_length,
            labels_padding_type=self.config.labels_padding_type,
            labels_max_length=self.config.labels_max_length,
        )

    def _load(self, split):
        data = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir)
        data = data.cast_column(self.config.audio_column, Audio(sampling_rate=self.config.sampling_rate))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_dict = self.data[index]
        transcript = sample_dict[self.config.transcript_column]
        audio_array = sample_dict[self.config.audio_column][self.config.audio_array_column]

        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt"
        )["input_features"]

        labels = self.tokenizer(
            transcript,
            padding_strategy=self.config.labels_padding_type,
            max_length=self.config.labels_max_length,
            return_tensors="pt",
        )

        return {
            "input_features": input_features,
            "labels": labels["token_ids"],
            "attention_mask": labels["attention_mask"],
        }
