"""
General classes for different data types.
"""
import os
from typing import List, Union, Literal

import PIL
import numpy as np

from ..preprocessors import Preprocessor

__all__ = [
    'Text',
    'Image',
]


class Text:
    def __init__(self, text: List[str]):
        self.text = text

    def __str__(self):
        return f"Sentence(text={self.text})"

    def process(self, preprocessors: List[Preprocessor], **kwargs):
        for p in preprocessors:
            self.text = p(self.text, **kwargs)
        return self.text


class Image:
    def __init__(self, path: Union[os.PathLike, str], lib: Literal["np", "torch", "tf"]):
        self.path = path
        self.lib = lib
        self.pil_image = PIL.Image.open(self.path).convert("RGB")
        self.np_image = np.array(self.pil_image)
        self._image = self.load()

    def __str__(self):
        return f"<Hezar.Image> size: {self.pil_image.width}x{self.pil_image.height}, mode: {self.pil_image.mode}"

    def load(self):
        if self.lib == "np":
            return self.np_image
        elif self.lib == "pil":
            return self.pil_image
        elif self.lib == "torch":
            import torch

            return torch.from_numpy(self.np_image)
        elif self.lib == "tf":
            import tensorflow as tf

            return tf.convert_to_tensor(self.np_image, dtype=tf.float32)
        else:
            raise ValueError(f"Invalid value for `lib` : `{self.lib}`!")

    @property
    def image(self):
        return self._image
