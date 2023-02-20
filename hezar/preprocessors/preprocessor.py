from hezar.utils import merge_kwargs_into_config


class Preprocessor:
    """
    Base class for all data preprocessors.

    Args:
        config: Preprocessor properties
    """

    def __init__(self, config, **kwargs):
        self.config = config.update(**kwargs)

    def __call__(self, inputs, **kwargs):
        """
        An abstract call method for a preprocessor. All preprocessors must implement this.

        Args:
            inputs: Raw inputs to process. Usually a list or a dict
            args: Extra arguments depending on the preprocessor
            kwargs: Extra keyword arguments depending on the preprocessor
        """
        raise NotImplementedError
