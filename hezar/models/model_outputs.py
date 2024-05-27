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
]


@dataclass
class ModelOutput:
    """
    Base class for all models' prediction outputs (`model.predict()`/`model.post_process()` outputs).

    Note that prediction outputs must all be a list of `ModelOutput` objects since we consider only batch inferences.

    The helper functions in the class enable it to be treated as a mapping or a dict object.
    """

    def dict(self):
        return asdict(self)

    def __str__(self):
        return str({k: v for k, v in self.dict().items() if v is not None})

    def __repr__(self):
        return str(self)

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


@dataclass(repr=False)
class MaskFillingOutput(ModelOutput):
    token: Optional[int] = None
    sequence: Optional[str] = None
    token_id: Optional[str] = None
    score: Optional[float] = None


@dataclass(repr=False)
class TextClassificationOutput(ModelOutput):
    label: Optional[str] = None
    score: Optional[float] = None


@dataclass(repr=False)
class SequenceLabelingOutput(ModelOutput):
    token: Optional[List[List[str]]] = None
    label: Optional[List[List[str]]] = None
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[List[List[float]]] = None


@dataclass(repr=False)
class TextGenerationOutput(ModelOutput):
    text: Optional[str] = None


@dataclass(repr=False)
class SpeechRecognitionOutput(ModelOutput):
    text: Optional[str] = None
    chunks: Optional[List[Dict]] = None


@dataclass(repr=False)
class Image2TextOutput(ModelOutput):
    text: Optional[str] = None
    score: Optional[str] = None
