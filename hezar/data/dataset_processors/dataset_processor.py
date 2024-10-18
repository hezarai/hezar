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


class DatasetProcessor:
    def __init__(self, batched=False, *args, **kwargs):
        """
        Base constructor that accepts a `batched` flag and any other arguments for child class initialization.

        Args:
            batched (bool): Whether to process data in batches or not.
        """
        self.batched = batched
        self.args = args
        self.kwargs = kwargs

    def __call__(self, examples, **fn_kwargs):
        """
        Method called when using the map function.
        Decides whether to call `process()` or `batch_process()` based on the `batched` flag.

        Args:
            examples (dict or list of dict): Data to process.
            **fn_kwargs: Additional keyword arguments passed through the `map` function as `fn_kwargs`.
                         For example, `fn_kwargs` can contain custom settings like `sampling_rate`.
        """
        if self.batched:
            return self.batch_process(examples, **fn_kwargs)
        else:
            return self.process(examples, **fn_kwargs)

    def process(self, example, **kwargs):
        """
        Method to process a single example
        """
        raise NotImplementedError

    def batch_process(self, examples, **kwargs):
        """
        Method to process a batch of examples.
        """
        raise NotImplementedError
