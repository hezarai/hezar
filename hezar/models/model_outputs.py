"""
Define all model outputs here
"""
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch


@dataclass
class ModelOutputs:
    """
    Base class for all model outputs (named based on tasks)

    The helper functions in the class enable it to be treated as a mapping or a dict object.
    """

    def dict(self):
        return asdict(self)

    def __getitem__(self, item):
        try:
            return self.dict()[item]
        except KeyError:
            raise AttributeError(f"`{self.__class__.__name__}` has no attribute `{item}`!")

    def __len__(self):
        return len(self.dict())

    def __iter__(self):
        return iter(self.dict())

    def keys(self):
        return list(self.dict().keys())

    def values(self):
        return list(self.dict().values())

    def items(self):
        return self.dict().items()


@dataclass
class LanguageModelingOutputs(ModelOutputs):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[torch.FloatTensor] = None


@dataclass
class TextClassificationOutputs(ModelOutputs):
    labels: Optional[List[str]] = None
    probs: Optional[List[float]] = None


@dataclass
class SequenceLabelingOutputs(ModelOutputs):
    tokens: Optional[List[List[str]]] = None
    tags: Optional[List[List[str]]] = None
    probs: Optional[List[List[float]]] = None


@dataclass
class Text2TextOutputs(ModelOutputs):
    generated_texts: Optional[List[str]] = None


@dataclass
class SpeechRecognitionOutputs(ModelOutputs):
    transcription: Optional[List[str]] = None
