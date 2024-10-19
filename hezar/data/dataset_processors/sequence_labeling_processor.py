import torch

from .dataset_processor import DatasetProcessor


class SequenceLabelingDatasetProcessor(DatasetProcessor):
    def __init__(self, tokenizer, label_all_tokens=True, ignore_index=-100, max_length=None, padding=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.padding = padding

    def _tokenize_and_align(self, tokens, labels, padding=None, max_length=None):
        """
        Tokenize and align tokens and labels for sequence labeling tasks.

        Args:
            tokens: List of tokens (for single examples) or list of lists (for batches).
            labels: List of labels (for single examples) or list of lists (for batches).
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
            return_tensors="torch"
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

        tokenized_inputs["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        return tokenized_inputs

    def process_single(self, data, padding=None, max_length=None):
        """
        Process a single example of sequence labeling data.

        Args:
            data: A single data example containing tokens and labels.
            padding: Padding strategy.
            max_length: Maximum sequence length.

        Returns:
            dict: Tokenized and aligned input data.
        """
        tokens = data["tokens"]
        labels = data["pos_tags"]

        tokenized_inputs = self._tokenize_and_align([tokens], [labels], padding=padding, max_length=max_length)

        data.update(tokenized_inputs)

        return data

    def process_batch(self, data, padding=None, max_length=None):
        """
        Process a batch of sequence labeling examples.

        Args:
            data: A batch of examples, containing tokens and labels.
            padding: Padding strategy.
            max_length: Maximum sequence length.

        Returns:
            dict: Tokenized and aligned batch data.
        """
        tokens = data["tokens"]
        labels = data["pos_tags"]

        tokenized_inputs = self._tokenize_and_align(tokens, labels, padding=padding, max_length=max_length)

        data.update(tokenized_inputs)

        return data
