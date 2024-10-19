from ...utils import shift_tokens_right
from .dataset_processor import DatasetProcessor


class ImageCaptioningDatasetProcessor(DatasetProcessor):
    def __init__(self, image_processor, tokenizer, max_length=None, padding=None):
        super().__init__()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def process_single(self, data, padding=None, max_length=None):
        """
        Process image and tokenize captions for a single data sample.

        Args:
            data: A data example containing the image and its caption
            padding: Padding type e.g, max_length, longest.
            max_length: Max length value if padding is set to max_length or the labels must be truncated.

        Returns:
            A dict of pixel values tensor of the processed image and labels token ids and attention mask.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        path = data["image_path"]
        text = data["label"]

        tokenized_inputs = self.tokenizer(text, padding=padding, max_length=max_length, return_tensors="torch")

        data["pixel_values"] = self.image_processor(path, return_tensors="torch")["pixel_values"]
        data["labels"] = tokenized_inputs["token_ids"]
        data["attention_mask"] = tokenized_inputs["attention_mask"]
        data["decoder_input_ids"] = shift_tokens_right(
            data["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

        return data

    def process_batch(self, data, padding=None, max_length=None):
        """
        Process image and tokenize captions for a batch of data samples.

        Args:
            data: A batch of data examples containing the images and their captions
            padding: Padding type e.g, max_length, longest.
            max_length: Max length value if padding is set to max_length or the labels must be truncated.

        Returns:
            A dict of pixel values tensor of the processed images and labels token ids and attention masks.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        paths = data["image_path"]
        texts = data["label"]

        tokenized_inputs = self.tokenizer(texts, padding=padding, max_length=max_length, return_tensors="torch")

        data["pixel_values"] = self.image_processor(paths, return_tensors="torch")["pixel_values"]
        data["labels"] = tokenized_inputs["token_ids"]
        data["attention_mask"] = tokenized_inputs["attention_mask"]
        data["decoder_input_ids"] = shift_tokens_right(
            data["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

        return data
