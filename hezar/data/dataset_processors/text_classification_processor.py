import torch

from .dataset_processor import DatasetProcessor


class TextClassificationDatasetProcessor(DatasetProcessor):
    def __init__(self, tokenizer, id2label, batched=False, max_length=None, padding=None):
        super().__init__(batched=batched)
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.padding = padding
        self.max_length = max_length

    def process(self, example, padding=None, max_length=None):
        """
        Process a single example.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        text = example["text"]
        label = example["label"]

        inputs = self.tokenizer(
            text,
            return_tensors="torch",
            truncation="longest_first",
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
        )
        example.update(inputs)
        example["labels"] = torch.tensor([label], dtype=torch.long)

        return example

    def batch_process(self, examples, padding=None, max_length=None):
        """
        Process a batch of examples.
        """
        padding = padding or self.padding
        max_length = max_length or self.max_length

        texts = examples["text"]
        labels = examples["label"]

        inputs = self.tokenizer(
            texts,
            return_tensors="torch",
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
        )
        examples.update(inputs)
        examples["labels"] = torch.tensor(labels, dtype=torch.long)

        return examples
