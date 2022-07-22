"""
General classes for different data types.
"""
import os
from typing import *

import PIL
import numpy as np


class Sentence:
    def __init__(self, text: str, language=None, normalizer=None, tokenizer=None):
        self.text = text
        self.language = language
        self.normalizer = normalizer
        self.tokenizer = tokenizer

    def tokenize(self, **kwargs):
        if self.tokenizer is None:
            raise ValueError(f'There was no tokenizer provided for this sentence!')
        inputs = self.tokenizer(self.text, **kwargs)
        return inputs

    def normalize(self):
        # TODO: implement normalization
        return Sentence(self.text, self.language, self.tokenizer)

    def filter_out(self, filter_list: List):
        for item in filter_list:
            self.text.replace(item, '')
        return Sentence(self.text, self.language, self.normalizer, self.tokenizer)


class Image:
    def __init__(self, path: Union[os.PathLike, str], lib: Literal['np', 'torch', 'tf']):
        self.path = path
        self.lib = lib
        self.pil_image = PIL.Image.open(self.path).convert('RGB')
        self.np_image = np.array(self.pil_image)
        self._image = self.load()

    def __str__(self):
        return f'<Hezar.Image> size: {self.pil_image.width}x{self.pil_image.height}, mode: {self.pil_image.mode}'

    def load(self):
        if self.lib == 'np':
            return self.np_image
        elif self.lib == 'pil':
            return self.pil_image
        elif self.lib == 'torch':
            import torch
            return torch.from_numpy(self.np_image)
        elif self.lib == 'tf':
            import tensorflow as tf
            return tf.convert_to_tensor(self.np_image, dtype=tf.float32)
        else:
            raise ValueError(f'Invalid value for `lib` : `{self.lib}`!')

    @property
    def image(self):
        return self._image
