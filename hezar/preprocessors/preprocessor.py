from typing import Dict, List, Tuple, Union

from ..builders import build_preprocessor
from ..constants import DEFAULT_PREPROCESSOR_SUBFOLDER


class Preprocessor:
    """
    Base class for all data preprocessors.

    Args:
        config: Preprocessor properties
    """

    preprocessor_subfolder = DEFAULT_PREPROCESSOR_SUBFOLDER

    def __init__(self, config, **kwargs):
        self.config = config.update(kwargs)

    def __call__(self, inputs, *args, **kwargs):
        """
        An abstract call method for a preprocessor. All preprocessors must implement this.

        Args:
            inputs: Raw inputs to process. Usually a list or a dict
            args: Extra arguments depending on the preprocessor
            **kwargs: Extra keyword arguments depending on the preprocessor
        """
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def push_to_hub(self, hub_path):
        raise NotImplementedError


class Sequential:
    """
    A sequence of preprocessors
    """
    def __init__(self, preprocessors: Union[List[Preprocessor], List[Tuple[str, Dict]]]):
        self._processors = self._prepare_processors(preprocessors)

    def __str__(self):
        return f"{self.__class__.__name__}({[p.config.name for p in self._processors] if self._processors else []})"

    @staticmethod
    def _prepare_processors(preprocessors):
        processors = []
        if isinstance(preprocessors, list) and len(preprocessors):
            for p in preprocessors:
                if isinstance(p, tuple):
                    name, kwargs = p
                    processors.append(build_preprocessor(name, **kwargs))
                elif isinstance(p, Preprocessor):
                    processors.append(p)
                else:
                    raise ValueError(f"Items in the preprocessors parameter must be either a `Preprocessor` or a tuple "
                                     f"of `(preprocessor_name, preprocessor_kwargs)`! Got `{type(p)}`!")
            return processors

    def __call__(self, inputs, *args, **kwargs):
        for processor in self._processors:
            inputs = processor(inputs, **kwargs)
        return inputs
