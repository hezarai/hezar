"""
Define all model outputs here
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


__all__ = [
    "ModelOutput",
    "MaskFillingOutput",
    "TextClassificationOutput",
    "SequenceLabelingOutput",
    "TextGenerationOutput",
    "SpeechRecognitionOutput",
    "Image2TextOutput",
    "TextDetectionOutput",
]


@dataclass
class ModelOutput:
    """
    Base class for all models' prediction outputs (`model.predict()`/`model.post_process()` outputs).

    Note that prediction outputs must all be a list of `ModelOutput` objects since we consider only batch inferences.

    The helper functions in the class enable it to be treated as a mapping or a dict object.
    """

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return str({k: v for k, v in self.to_dict().items() if v is not None})

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        try:
            return self.to_dict()[item]
        except KeyError:
            raise AttributeError(f"`{self.__class__.__name__}` has no attribute `{item}`!")

    def __len__(self):
        return len(self.to_dict())

    def __iter__(self):
        return iter(self.to_dict())

    def keys(self):
        return list(self.to_dict().keys())

    def values(self):
        return list(self.to_dict().values())

    def items(self):
        return self.to_dict().items()


@dataclass(repr=False)
class MaskFillingOutput(ModelOutput):
    token: Optional[int] = None
    sequence: str | None = None
    token_id: str | None = None
    score: Optional[float] = None


@dataclass(repr=False)
class TextClassificationOutput(ModelOutput):
    label: str | None = None
    score: Optional[float] = None


@dataclass(repr=False)
class SequenceLabelingOutput(ModelOutput):
    token: Optional[list[list[str]]] = None
    label: Optional[list[list[str]]] = None
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[list[list[float]]] = None


@dataclass(repr=False)
class TextGenerationOutput(ModelOutput):
    text: str | None = None


@dataclass(repr=False)
class SpeechRecognitionOutput(ModelOutput):
    text: str | None = None
    chunks: Optional[list[dict]] = None


@dataclass(repr=False)
class Image2TextOutput(ModelOutput):
    text: str | None = None
    score: str | None = None


@dataclass(repr=False)
class TextDetectionOutput(ModelOutput):
    boxes: list[int] | None = None
