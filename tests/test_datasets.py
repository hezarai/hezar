from typing import Dict
from torch.utils.data import DataLoader
from hezar.data import Dataset
import pytest

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

TASK_TO_REQUIRED_FIELDS = {
    "text-classification": ["token_ids", "attention_mask", "labels"],
    "sequence-labeling": ["token_ids", "attention_mask", "word_ids", "labels"],
    "ocr": ["pixel_values", "labels"],
    "text-summarization": ["token_ids", "attention_mask", "labels"],
}

INVALID_DATASET_TYPE = "Dataset instance must be of type `Dataset`, got `{}`!"
INVALID_BATCH_TYPE = "A data batch must be a dictionary not `{}`!"
INVALID_DATASET_FIELDS = "Dataset samples must have the field `{}`!"


def create_dataloader(dataset, batch_size, shuffle, collate_fn):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


@pytest.mark.parametrize("task", TASK_TO_HUB_MAPPING.keys())
def test_load_dataset(task):
    required_fields = TASK_TO_REQUIRED_FIELDS[task]

    train_dataset = Dataset.load(
        TASK_TO_HUB_MAPPING[task],
        split="train",
        tokenizer_path=TASK_TO_TOKENIZER_MAPPING[task]
    )
    assert isinstance(train_dataset, Dataset), INVALID_DATASET_TYPE.format(type(train_dataset))
    sample = train_dataset[0]
    assert isinstance(sample, Dict)
    for field in required_fields:
        assert field in sample, INVALID_DATASET_FIELDS.format(field)

    test_dataset = Dataset.load(
        TASK_TO_HUB_MAPPING[task],
        split="test",
        tokenizer_path=TASK_TO_TOKENIZER_MAPPING[task]
    )
    assert isinstance(test_dataset, Dataset), INVALID_DATASET_TYPE.format(type(test_dataset))
    sample = test_dataset[0]
    assert isinstance(sample, Dict)
    for field in required_fields:
        assert field in sample, INVALID_DATASET_FIELDS.format(field)

    # Dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=train_dataset.data_collator,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=test_dataset.data_collator,
    )
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    assert isinstance(train_batch, Dict), INVALID_BATCH_TYPE.format(type(train_batch))
    assert isinstance(test_batch, Dict), INVALID_BATCH_TYPE.format(type(test_batch))
