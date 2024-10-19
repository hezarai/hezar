"""
Dataset processors are a bunch of callable classes to be passed to be used as map functions for any dataset on the Hub.
Note that the main dataset classes are already implemented in a way that the processing is done in the `__getitem__`
method and these classes are only used for when the dataset has been loaded using HuggingFace datasets library.

Example:
>>> from datasets import load_dataset
>>> from hezar.data import SpeechRecognitionDatasetProcessor

>>> data_processor = SpeechRecognitionDatasetProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
>>> dataset = load_dataset("hezarai/common-voice-13-fa")
>>> dataset = dataset.map(data_processor, batched=True, batch_size=1000)

"""
from ...constants import Backends
from ...utils import is_backend_available, verify_dependencies


if is_backend_available(Backends.DATASETS):
    from datasets.formatting.formatting import LazyBatch, LazyRow


class DatasetProcessor:
    required_backends = [Backends.DATASETS]

    def __init__(self, *args, **kwargs):
        """
        Base constructor that accepts a `batched` flag and any other arguments for child class initialization.
        """
        verify_dependencies(self, self.required_backends)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: LazyBatch | LazyRow, **kwargs):
        """
        Method called when using the map function.
        Decides whether to call `process_single()` or `process_batch()` based on the data values.

        Args:
            data: A dict of feature name -> sample or batch of samples mapping.
            **kwargs: Additional keyword arguments passed through the `map` function as `kwargs`.
        """
        if isinstance(data, LazyRow):
            return self.process_single(data, **kwargs)
        elif isinstance(data, LazyBatch):
            return self.process_batch(data, **kwargs)
        else:
            raise ValueError(f"The input data must be either `LazyBatch` or `LazyRow`, got `{type(data)}`!")

    def process_single(self, data: LazyRow, **kwargs):
        """
        Method to process a single example
        """
        raise NotImplementedError

    def process_batch(self, data: LazyBatch, **kwargs):
        """
        Method to process a batch of examples.
        """
        raise NotImplementedError
