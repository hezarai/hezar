from ...utils import shift_tokens_right
from .dataset_processor import DatasetProcessor


class ImageCaptioningDatasetProcessor(DatasetProcessor):
    def __init__(self, image_processor, tokenizer, batched=False, max_length=None, padding=None):
        super().__init__(batched=batched)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def process(self, example, padding=None, max_length=None):
        """
        Process image and tokenize captions for a single data sample.

        Args:
            example: A data example containing the image and its caption
            padding: Padding type e.g, max_length, longest.
            max_length: Max length value if padding is set to max_length or the labels must be truncated.

        Returns:
            A dict of pixel values tensor of the processed image and labels token ids and attention mask.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        path, text = example["path"], example["text"]
        tokenized_inputs = self.tokenizer(text, padding=padding, max_length=max_length, return_tensors="pt")

        example["pixel_values"] = self.image_processor(path, return_tensors="pt")["pixel_values"]
        example["labels"] = tokenized_inputs["input_ids"]
        example["attention_mask"] = tokenized_inputs["attention_mask"]
        example["decoder_input_ids"] = shift_tokens_right(
            example["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

        return example

    def batch_process(self, examples, padding=None, max_length=None):
        """
        Process image and tokenize captions for a single data sample.

        Args:
            examples: A batch of data examples containing the images and their captions
            padding: Padding type e.g, max_length, longest.
            max_length: Max length value if padding is set to max_length or the labels must be truncated.

        Returns:
            A dict of pixel values tensor of the processed images and labels token ids and attention masks.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        paths = [example["path"] for example in examples]
        texts = [example["text"] for example in examples]

        tokenized_inputs = self.tokenizer(texts, padding=padding, max_length=max_length, return_tensors="pt")

        examples["pixel_values"] = self.image_processor(paths, return_tensors="pt")["pixel_values"]
        examples["labels"] = tokenized_inputs["input_ids"]
        examples["attention_mask"] = tokenized_inputs["attention_mask"]
        examples["decoder_input_ids"] = shift_tokens_right(
            examples["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

        return examples
