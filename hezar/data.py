"""
General classes for different data types.
"""
import os
from typing import Union, Optional, Literal, List
from dataclasses import dataclass, field


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
    def __init__(self, path: Union[os.PathLike, str], lib: Literal['opencv', 'pil'] = 'opencv') -> None:
        self.path = path
        self.lib = lib

    @classmethod
    def load(cls, path, lib: LIB):
        if lib == 'opencv':
            import cv2
            image = cv2.imread(path)
        elif lib == 'pil':
            import PIL.Image as Image
            image = Image.open(path).convert('RGB')
        else:
            raise ValueError(f'Invalid value for `lib` : {lib}, expected {cls.lib}')

        return image
