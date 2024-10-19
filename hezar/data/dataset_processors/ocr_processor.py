import torch

from ...utils import reverse_string_digits
from .dataset_processor import DatasetProcessor


class OCRDatasetProcessor(DatasetProcessor):
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

    def _text_to_tensor(self, text):
        """
        Convert text to tensor based on the configured text_split_type.

        Args:
            text (str): The raw text.

        Returns:
            torch.Tensor: The output tensor.

        """
        if self.text_split_type == "tokenize":
            token_ids = self.tokenizer(text, padding="max_length", max_length=self.max_length)["input_ids"]
            token_ids = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in token_ids]
            labels = torch.tensor(token_ids)
        elif self.text_split_type == "char_split":
            if self.reverse_digits:
                text = reverse_string_digits(text)
            label2id = {v: k for k, v in self.id2label.items()}
            labels = [label2id[char] for char in text]
            labels = torch.LongTensor(labels)
        else:
            raise ValueError(f"Invalid `text_split_type={self.text_split_type}`")
        return labels

    def process_single(self, data):
        """
        Process a single image-to-text OCR example.

        Args:
            data: A data example containing an image path and corresponding text.

        Returns:
            dict: Processed inputs with pixel values and text labels.
        """
        path = data[self.image_field]
        text = data[self.text_field]
        pixel_values = self.image_processor(path, return_tensors="torch")["pixel_values"][0]
        labels = self._text_to_tensor(text)
        return {"pixel_values": pixel_values, "labels": labels}

    def process_batch(self, data):
        """
        Process a batch of image-to-text OCR examples.

        Args:
            data: A batch of data examples containing image paths and corresponding texts.

        Returns:
            dict: Batch of processed inputs with pixel values and text labels.
        """
        paths = data[self.image_field]
        texts = data[self.text_field]

        # Process images in batch
        pixel_values = self.image_processor(paths, return_tensors="torch")["pixel_values"]

        # Process text labels in batch
        labels = []
        for text in texts:
            labels.append(self._text_to_tensor(text))

        return {"pixel_values": pixel_values, "labels": labels}
