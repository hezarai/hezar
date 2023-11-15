from unittest import TestCase
from typing import Dict

from torch.utils.data import DataLoader
from hezar.data import Dataset

TASK_TO_HUB_MAPPING = {
    "text-classification": "hezarai/sentiment-dksf",
    "sequence-labeling": "hezarai/lscp-pos-500k",
    "ocr": "hezarai/persian-license-plate-v1",
    "text-summarization": "hezarai/xlsum-fa",
}
TASK_TO_TOKENIZER_MAPPING = {
    "text-classification": "hezarai/bert-base-fa",
    "sequence-labeling": "hezarai/bert-base-fa",
    "ocr": "hezarai/crnn-base-fa-64x256",
    "text-summarization": "hezarai/t5-base-fa",
}

INVALID_DATASET_TYPE = "Dataset instance must be of type `Dataset`, got `{}`!"
INVALID_BATCH_TYPE = "A data batch must be a dictionary not `{}`!"
INVALID_DATASET_FIELDS = "Dataset samples must have the field `{}`!"


class HubDatasetsTestCase(TestCase):
    @staticmethod
    def _create_dataloader(dataset, batch_size, shuffle, collate_fn):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

    def test_load_text_classification(self):
        required_fields = ["token_ids", "attention_mask", "labels"]

        train_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["text-classification"],
            split="train",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["text-classification"]
        )
        self.assertIsInstance(train_dataset, Dataset, INVALID_DATASET_TYPE.format(type(train_dataset)))
        sample = train_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        test_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["text-classification"],
            split="test",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["text-classification"]
        )
        self.assertIsInstance(test_dataset, Dataset, INVALID_DATASET_TYPE.format(type(test_dataset)))
        sample = test_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        # Dataloaders
        train_loader = self._create_dataloader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=train_dataset.data_collator,
        )
        test_loader = self._create_dataloader(
            test_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=test_dataset.data_collator,
        )
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        self.assertIsInstance(train_batch, Dict, INVALID_BATCH_TYPE.format(type(train_batch)))
        self.assertIsInstance(test_batch, Dict, INVALID_BATCH_TYPE.format(type(test_batch)))

    def test_load_sequence_labeling(self):
        required_fields = ["token_ids", "attention_mask", "special_tokens_mask", "word_ids", "labels"]

        train_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["sequence-labeling"],
            split="train",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["sequence-labeling"]
        )
        self.assertIsInstance(train_dataset, Dataset, INVALID_DATASET_TYPE.format(type(train_dataset)))
        sample = train_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        test_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["sequence-labeling"],
            split="test",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["sequence-labeling"]
        )
        self.assertIsInstance(test_dataset, Dataset, INVALID_DATASET_TYPE.format(type(test_dataset)))
        sample = test_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        # Dataloaders
        train_loader = self._create_dataloader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=train_dataset.data_collator,
        )
        test_loader = self._create_dataloader(
            test_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=test_dataset.data_collator,
        )
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        self.assertIsInstance(train_batch, Dict, INVALID_BATCH_TYPE.format(type(train_batch)))
        self.assertIsInstance(test_batch, Dict, INVALID_BATCH_TYPE.format(type(test_batch)))

    def test_load_ocr(self):
        required_fields = ["pixel_values", "labels"]

        train_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["ocr"],
            split="train",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["ocr"]
        )
        self.assertIsInstance(train_dataset, Dataset, INVALID_DATASET_TYPE.format(type(train_dataset)))
        sample = train_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        test_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["ocr"],
            split="test",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["ocr"]
        )
        self.assertIsInstance(test_dataset, Dataset, INVALID_DATASET_TYPE.format(type(test_dataset)))
        sample = test_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        # Dataloaders
        train_loader = self._create_dataloader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=train_dataset.data_collator,
        )
        test_loader = self._create_dataloader(
            test_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=test_dataset.data_collator,
        )
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        self.assertIsInstance(train_batch, Dict, INVALID_BATCH_TYPE.format(type(train_batch)))
        self.assertIsInstance(test_batch, Dict, INVALID_BATCH_TYPE.format(type(test_batch)))

    def test_load_text_summarization(self):
        required_fields = ["token_ids", "attention_mask", "labels"]

        train_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["text-summarization"],
            split="train",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["text-summarization"]
        )
        self.assertIsInstance(train_dataset, Dataset, INVALID_DATASET_TYPE.format(type(train_dataset)))
        sample = train_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        test_dataset = Dataset.load(
            TASK_TO_HUB_MAPPING["text-summarization"],
            split="test",
            tokenizer_path=TASK_TO_TOKENIZER_MAPPING["text-summarization"]
        )
        self.assertIsInstance(test_dataset, Dataset, INVALID_DATASET_TYPE.format(type(test_dataset)))
        sample = test_dataset[0]
        self.assertIsInstance(sample, Dict)
        [self.assertIn(x, sample, INVALID_DATASET_FIELDS.format(x)) for x in required_fields]

        # Dataloaders
        train_loader = self._create_dataloader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=train_dataset.data_collator,
        )
        test_loader = self._create_dataloader(
            test_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=test_dataset.data_collator,
        )
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        self.assertIsInstance(train_batch, Dict, INVALID_BATCH_TYPE.format(type(train_batch)))
        self.assertIsInstance(test_batch, Dict, INVALID_BATCH_TYPE.format(type(test_batch)))
