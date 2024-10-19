import torch

from .dataset_processor import DatasetProcessor


class TextClassificationDatasetProcessor(DatasetProcessor):
    def __init__(self, tokenizer, max_length=None, padding=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def process_single(self, data, padding=None, max_length=None):
        """
        Process a single example.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        text = data["text"]
        label = data["label"]

        inputs = self.tokenizer(
            text,
            return_tensors="torch",
            truncation="longest_first",
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
        )
        data.update(inputs)
        data["labels"] = torch.tensor([label], dtype=torch.long)

        return data

    def process_batch(self, data, padding=None, max_length=None):
        """
        Process a batch of examples.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        texts = data["text"]
        labels = data["label"]

        inputs = self.tokenizer(
            texts,
            return_tensors="torch",
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
        )
        data.update(inputs)
        data["labels"] = torch.tensor(labels, dtype=torch.long)

        return data
