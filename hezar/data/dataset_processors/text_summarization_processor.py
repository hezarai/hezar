from .dataset_processor import DatasetProcessor


class TextSummarizationDatasetProcessor(DatasetProcessor):
    def __init__(
        self,
        tokenizer,
        prefix=None,
        max_length=None,
        labels_max_length=None,
        text_field="text",
        summary_field="summary",
        padding="longest",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_length = max_length
        self.labels_max_length = labels_max_length
        self.text_field = text_field
        self.summary_field = summary_field
        self.padding = padding

    def process_single(self, data, padding=None, max_length=None, labels_max_length=None):
        """
        Process a single example for text summarization.

        Args:
            data: A data example containing text and summary.
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
            return_tensors="torch",
            max_length=max_length,
            padding=padding,
            return_attention_mask=True,
            truncation=True
        )
        labels = self.tokenizer(
            summary,
            return_tensors="torch",
            max_length=labels_max_length,
            padding=padding,
            return_attention_mask=True,
            truncation=True
        )

        inputs["labels"] = labels["token_ids"].clone()

        return inputs

    def process_batch(self, data, padding=None, max_length=None, labels_max_length=None):
        """
        Process a batch of examples for text summarization.

        Args:
            data: A batch of examples containing texts and summaries.
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
            return_tensors="torch",
            max_length=max_length,
            padding=padding,
            return_attention_mask=True,
            truncation=True
        )
        labels = self.tokenizer(
            summaries,
            return_tensors="torch",
            max_length=labels_max_length,
            padding=padding,
            return_attention_mask=True,
            truncation=True
        )

        inputs["labels"] = labels["token_ids"].clone()

        return inputs
