"""
Dataset processors are a bunch of callable classes to be passed as map functions for any dataset on the Hub.
Note that the main dataset classes are already implemented in a way that the processing is done in the `__getitem__`
method and these classes are only used for when the dataset has been loaded using the HuggingFace datasets library,
and you want to get advantage of the multiprocessing/batch processing/caching functionalities of the HF datasets.

Example:
>>> from datasets import load_dataset
>>> from hezar.data import SpeechRecognitionDatasetProcessor

>>> data_processor = SpeechRecognitionDatasetProcessor(feature_extractor=feature_extractor,tokenizer=tokenizer)
>>> dataset = load_dataset("hezarai/common-voice-13-fa")
>>> dataset = dataset.map(data_processor, batched=True, batch_size=1000)
"""

import torch

from ..constants import Backends
from ..utils import is_backend_available, reverse_string_digits, verify_dependencies


if is_backend_available(Backends.DATASETS):
    from datasets.formatting.formatting import LazyBatch, LazyRow

__all__ = [
    "DatasetProcessor",
    "ImageCaptioningDatasetProcessor",
    "OCRDatasetProcessor",
    "SequenceLabelingDatasetProcessor",
    "SpeechRecognitionDatasetProcessor",
    "TextClassificationDatasetProcessor",
    "TextSummarizationDatasetProcessor",
]


class DatasetProcessor:
    """
    The base callable dataset processor class that can handle both single and batched mode dataset mapping.
    """
    required_backends = [Backends.DATASETS]

    def __init__(self, *args, **kwargs):
        verify_dependencies(self, self.required_backends)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: LazyBatch | LazyRow, return_tensors="list", **kwargs):
        """
        Method called when using the map function.
        Decides whether to call `process_single()` or `process_batch()` based on the data values.

        Args:
            data: A dict of feature name -> sample or batch of samples mapping.
            return_tensors: The type of the returning tensors (list, torch, numpy)
            **kwargs: Additional keyword arguments passed through the `map` function as `kwargs`.
        """
        if isinstance(data, LazyRow):
            return self.process_single(data, return_tensors=return_tensors, **kwargs)
        elif isinstance(data, LazyBatch):
            return self.process_batch(data, return_tensors=return_tensors, **kwargs)
        else:
            raise ValueError(f"The input data must be either `LazyBatch` or `LazyRow`, got `{type(data)}`!")

    def process_single(self, data: LazyRow, return_tensors=None, **kwargs):
        """
        Process a single data example.

        Args:
            data: A data sample dict
            return_tensors: The type of the returning tensors (list, torch, numpy)
            **kwargs: Additional arguments

        Returns:
            The updated data dict
        """
        raise NotImplementedError

    def process_batch(self, data: LazyBatch, return_tensors=None, **kwargs):
        """
        Process a batch of data examples.

        Args:
            data: A data sample dict
            return_tensors: The type of the returning tensors (list, torch, numpy)
            **kwargs: Additional arguments

        Returns:
            The updated data dict
        """
        raise NotImplementedError


class ImageCaptioningDatasetProcessor(DatasetProcessor):
    """
    Dataset processor for image captioning datasets. This class handles tokenization and image processing.
    """

    def __init__(self, image_processor, tokenizer, max_length=None, padding=None):
        super().__init__()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    @staticmethod
    def _shift_tokens_right(input_ids: list[list[int]], pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        # Initialize shifted_input_ids as a list of lists with the same shape as input_ids
        shifted_input_ids = [[0] * len(row) for row in input_ids]

        for i, row in enumerate(input_ids):
            # Shift each row one token to the right
            shifted_input_ids[i][1:] = row[:-1]
            # Set the first token to decoder_start_token_id
            shifted_input_ids[i][0] = decoder_start_token_id

            # Replace any -100 values with pad_token_id
            shifted_input_ids[i] = [pad_token_id if token == -100 else token for token in shifted_input_ids[i]]

        return shifted_input_ids

    def process_single(self, data, return_tensors=None, padding=None, max_length=None):
        """
        Process image and tokenize captions for a single data sample.

        Args:
            data: A data example containing the image and its caption
            padding: Padding type e.g, max_length, longest.
            max_length: Max length value if padding is set to max_length or the labels must be truncated.
            return_tensors: The type of the returning tensors (list, torch, numpy)

        Returns:
            A dict of pixel values tensor of the processed image and labels token ids and attention mask.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        path = data["image_path"]
        text = data["label"]

        tokenized_inputs = self.tokenizer(text, padding=padding, max_length=max_length, return_tensors=return_tensors)

        data["pixel_values"] = self.image_processor(path, return_tensors=return_tensors)["pixel_values"]
        data["labels"] = tokenized_inputs["token_ids"]
        data["decoder_attention_mask"] = tokenized_inputs["attention_mask"]
        data["decoder_input_ids"] = self._shift_tokens_right(
            [data["labels"]],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )[0]

        return data

    def process_batch(self, data, return_tensors=None, padding=None, max_length=None):
        """
        Process image and tokenize captions for a batch of data samples.

        Args:
            data: A batch of data examples containing the images and their captions
            padding: Padding type e.g, max_length, longest.
            max_length: Max length value if padding is set to max_length or the labels must be truncated.
            return_tensors: The type of the returning tensors (list, torch, numpy)

        Returns:
            A dict of pixel values tensor of the processed images and labels token ids and attention masks.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        paths = data["image_path"]
        texts = data["label"]

        tokenized_inputs = self.tokenizer(texts, padding=padding, max_length=max_length, return_tensors=return_tensors)

        data["pixel_values"] = self.image_processor(paths, return_tensors=return_tensors)["pixel_values"]
        data["labels"] = tokenized_inputs["token_ids"]
        data["decoder_attention_mask"] = tokenized_inputs["attention_mask"]
        data["decoder_input_ids"] = self._shift_tokens_right(
            data["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

        return data


class OCRDatasetProcessor(DatasetProcessor):
    """
    Dataset processor class for OCR which can handle both tokenizer-based or character-split-based datasets.
    """

    def __init__(
        self,
        image_processor,
        tokenizer=None,
        text_split_type="char_split",
        max_length=None,
        reverse_digits=False,
        id2label=None,
        image_field="image_path",
        text_field="text",
    ):
        super().__init__()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.text_split_type = text_split_type
        self.max_length = max_length
        self.reverse_digits = reverse_digits
        self.id2label = id2label
        self.image_field = image_field
        self.text_field = text_field

    def _text_to_ids(self, text):
        """
        Convert text to tensor based on the configured text_split_type.

        Args:
            text (str): The raw text.

        Returns:
            torch.Tensor: The output tensor.

        """
        if self.text_split_type == "tokenize":
            token_ids = self.tokenizer(text, padding="max_length", max_length=self.max_length)["input_ids"]
            labels = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in token_ids]
        elif self.text_split_type == "char_split":
            if self.reverse_digits:
                text = reverse_string_digits(text)
            label2id = {v: k for k, v in self.id2label.items()}
            labels = [label2id[char] for char in text]
        else:
            raise ValueError(f"Invalid `text_split_type={self.text_split_type}`")
        return labels

    def process_single(self, data, return_tensors=None):
        """
        Process a single image-to-text OCR example.

        Args:
            data: A data example containing an image path and corresponding text.
            return_tensors: The type of the returning tensors (list, torch, numpy)

        Returns:
            dict: Processed inputs with pixel values and text labels.
        """
        path = data[self.image_field]
        text = data[self.text_field]
        pixel_values = self.image_processor(path, return_tensors=return_tensors)["pixel_values"][0]
        labels = self._text_to_ids(text)
        return {"pixel_values": pixel_values, "labels": labels}

    def process_batch(self, data, return_tensors=None):
        """
        Process a batch of image-to-text OCR examples.

        Args:
            data: A batch of data examples containing image paths and corresponding texts.
            return_tensors: The type of the returning tensors (list, torch, numpy)

        Returns:
            dict: Batch of processed inputs with pixel values and text labels.
        """
        paths = data[self.image_field]
        texts = data[self.text_field]

        # Process images in batch
        pixel_values = self.image_processor(paths, return_tensors=return_tensors)["pixel_values"]

        # Process text labels in batch
        labels = [self._text_to_ids(text) for text in texts]

        return {"pixel_values": pixel_values, "labels": labels}


class SequenceLabelingDatasetProcessor(DatasetProcessor):
    """
    Dataset processor class for sequence labeling datasets. Handles tokenization and label alignment.
    """

    def __init__(self, tokenizer, label_all_tokens=True, ignore_index=-100, max_length=None, padding=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.padding = padding

    def _tokenize_and_align(self, tokens, labels, return_tensors=None, padding=None, max_length=None):
        """
        Tokenize and align tokens and labels for sequence labeling tasks.

        Args:
            tokens: List of tokens (for single examples) or list of lists (for batches).
            labels: List of labels (for single examples) or list of lists (for batches).
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Padding strategy for tokenization.
            max_length: Maximum sequence length to truncate/pad.

        Returns:
            dict: Tokenized and aligned inputs with labels.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        # Tokenize and return word IDs for mapping labels to subword tokens
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_word_ids=True,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors
        )
        word_ids = tokenized_inputs["word_ids"]

        # Align labels with tokens
        aligned_labels = []
        for batch_idx, batch_word_ids in enumerate(word_ids):
            previous_word_idx = None
            label_ids = []
            for word_idx in batch_word_ids:
                # Assign ignore index for special tokens
                if word_idx is None:
                    label_ids.append(self.ignore_index)
                elif word_idx != previous_word_idx:
                    # Assign the label for the first token of each word
                    label_ids.append(labels[batch_idx][word_idx])
                else:
                    # Assign label for subword tokens (if label_all_tokens is True)
                    label_ids.append(labels[batch_idx][word_idx] if self.label_all_tokens else self.ignore_index)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    def process_single(self, data, return_tensors=None, padding=None, max_length=None):
        """
        Process a single example of sequence labeling data.

        Args:
            data: A single data example containing tokens and labels.
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Padding strategy.
            max_length: Maximum sequence length.

        Returns:
            dict: Tokenized and aligned input data.
        """
        tokens = data["tokens"]
        labels = data["pos_tags"]

        tokenized_inputs = self._tokenize_and_align(
            [tokens],
            [labels],
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
        )
        tokenized_inputs = {k: v[0] for k, v in tokenized_inputs.items()}
        data.update(tokenized_inputs)

        return data

    def process_batch(self, data, return_tensors=None, padding=None, max_length=None):
        """
        Process a batch of sequence labeling examples.

        Args:
            data: A batch of examples, containing tokens and labels.
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Padding strategy.
            max_length: Maximum sequence length.

        Returns:
            dict: Tokenized and aligned batch data.
        """
        tokens = data["tokens"]
        labels = data["pos_tags"]

        tokenized_inputs = self._tokenize_and_align(
            tokens,
            labels,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
        )

        data.update(tokenized_inputs)

        return data


class SpeechRecognitionDatasetProcessor(DatasetProcessor):
    """
    Processor class for speech recognition datasets. Handles audio feature extraction and labels tokenization.
    """

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        sampling_rate=16000,
        audio_array_padding=None,
        max_audio_array_length=None,
        labels_padding=None,
        labels_max_length=None,
        audio_column="audio",
        transcript_column="transcript",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.audio_array_padding = audio_array_padding
        self.max_audio_array_length = max_audio_array_length
        self.labels_padding = labels_padding
        self.labels_max_length = labels_max_length
        self.audio_column = audio_column
        self.transcript_column = transcript_column

    def process_single(self, data, return_tensors=None):
        """
        Process a single speech recognition example.

        Args:
            data: A data example containing audio and its transcript.
            return_tensors: The type of the returning tensors (list, torch, numpy)

        Returns:
            dict: Processed input features and labels.
        """
        audio_array = data[self.audio_column]["array"]
        transcript = data[self.transcript_column]

        # Extract input features from audio
        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.sampling_rate,
            padding=self.audio_array_padding,
            max_length=self.max_audio_array_length,
            return_tensors=return_tensors,
        )["input_features"]

        # Tokenize the transcript
        labels = self.tokenizer(
            transcript,
            padding=self.labels_padding,
            max_length=self.labels_max_length,
            return_tensors=return_tensors,
        )

        data["input_features"] = input_features
        data["labels"] = labels["token_ids"]
        data["attention_mask"] = labels["attention_mask"]

        return data

    def process_batch(self, data, return_tensors=None):
        """
        Process a batch of speech recognition examples.

        Args:
            data: A batch of data examples containing audio arrays and their corresponding transcripts.
            return_tensors: The type of the returning tensors (list, torch, numpy)

        Returns:
            dict: Batch of processed input features and labels.
        """
        audio_arrays = [x["array"] for x in data[self.audio_column]]
        transcripts = data[self.transcript_column]

        # Extract input features in batch
        input_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            padding=self.audio_array_padding,
            max_length=self.max_audio_array_length,
            return_tensors=return_tensors,
        )["input_features"]

        # Tokenize transcripts in batch
        labels = self.tokenizer(
            transcripts,
            padding=self.labels_padding,
            max_length=self.labels_max_length,
            return_tensors=return_tensors,
        )

        data["input_features"] = input_features
        data["labels"] = labels["token_ids"]
        data["attention_mask"] = labels["attention_mask"]

        return data


class TextClassificationDatasetProcessor(DatasetProcessor):
    """
    Processor class for text classification datasets. Handles tokenization of the texts.
    """

    def __init__(self, tokenizer, max_length=None, padding=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def process_single(self, data, return_tensors=None, padding=None, max_length=None):
        """
        Process a single example for text classification.

        Args:
            data: A single data example dict
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Token ids padding type
            max_length: Max input length

        Returns:
            The updated data dictionary
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        text = data["text"]
        label = data["label"]

        inputs = self.tokenizer(
            text,
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        data.update(inputs)
        data["labels"] = torch.tensor(label, dtype=torch.long)

        return data

    def process_batch(self, data, return_tensors=None, padding=None, max_length=None):
        """
        Process a batch of examples for text classification.

        Args:
            data: A single data example dict
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Token ids padding type
            max_length: Max input length

        Returns:
            The updated data dictionary
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        texts = data["text"]
        labels = data["label"]

        inputs = self.tokenizer(
            texts,
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        data.update(inputs)
        data["labels"] = torch.tensor(labels, dtype=torch.long)

        return data


class TextSummarizationDatasetProcessor(DatasetProcessor):
    """
    Processor class for text summarization datasets. Handles tokenization of the inputs and labels.
    """

    def __init__(
        self,
        tokenizer,
        prefix=None,
        max_length=None,
        labels_max_length=None,
        text_field="text",
        summary_field="summary",
        padding=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_length = max_length
        self.labels_max_length = labels_max_length
        self.text_field = text_field
        self.summary_field = summary_field
        self.padding = padding

    def process_single(self, data, return_tensors=None, padding=None, max_length=None, labels_max_length=None):
        """
        Process a single example for text summarization.

        Args:
            data: A data example containing text and summary.
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Padding strategy.
            max_length: Max length for input text.
            labels_max_length: Max length for summary labels.

        Returns:
            dict: Tokenized inputs and labels for summarization task.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length
        labels_max_length = labels_max_length or self.labels_max_length

        text = data[self.text_field]
        summary = data[self.summary_field]

        # Add prefix if needed for conditional generation
        if self.prefix is not None:
            text = self.prefix + text

        # Tokenize inputs and labels
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        labels = self.tokenizer(
            summary,
            max_length=labels_max_length,
            padding=padding,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        inputs["labels"] = labels["token_ids"]

        return inputs

    def process_batch(self, data, return_tensors=None, padding=None, max_length=None, labels_max_length=None):
        """
        Process a batch of examples for text summarization.

        Args:
            data: A batch of examples containing texts and summaries.
            return_tensors: The type of the returning tensors (list, torch, numpy)
            padding: Padding strategy.
            max_length: Max length for input texts.
            labels_max_length: Max length for summary labels.

        Returns:
            dict: Tokenized inputs and labels for summarization task.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length
        labels_max_length = labels_max_length or self.labels_max_length

        texts = data[self.text_field]
        summaries = data[self.summary_field]

        # Add prefix if needed for conditional generation
        if self.prefix is not None:
            texts = [self.prefix + text for text in texts]

        # Tokenize inputs and labels in batch
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        labels = self.tokenizer(
            summaries,
            max_length=labels_max_length,
            padding=padding,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        inputs["labels"] = labels["token_ids"]

        return inputs
