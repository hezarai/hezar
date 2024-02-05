import os
from typing import Dict

import pytest
from torch.utils.data import DataLoader

from hezar.data import Dataset
from hezar.utils import clean_cache

CI_MODE = os.environ.get("CI_MODE", "FALSE")

DATASETS_MAPPING = {
    "text-classification": {
        "path": "hezarai/sentiment-dksf",
        "tokenizer_path": "hezarai/bert-base-fa",
    },
    "sequence-labeling": {
        "path": "hezarai/lscp-pos-500k",
        "tokenizer_path": "hezarai/bert-base-fa",
    },
    "ocr": {
        "path": "hezarai/persian-license-plate-v1",
        "tokenizer_path": "hezarai/crnn-fa-printed-96-long",
    },
    "image-captioning": {
        "path": "hezarai/flickr30k-fa",
        "tokenizer_path": "hezarai/roberta-base-fa",
    },
    "text-summarization": {
        "path": "hezarai/xlsum-fa",
        "tokenizer_path": "hezarai/t5-base-fa",
    },
    "speech-recognition": {
        "path": "hezarai/common-voice-13-fa",
        "tokenizer_path": "hezarai/whisper-small-fa",
        "feature_extractor_path": "hezarai/whisper-small-fa"
    },
}

TASK_TO_REQUIRED_FIELDS = {
    "text-classification": ["token_ids", "attention_mask", "labels"],
    "sequence-labeling": ["token_ids", "attention_mask", "word_ids", "labels"],
    "ocr": ["pixel_values", "labels"],
    "image-captioning": ["pixel_values", "labels"],
    "text-summarization": ["token_ids", "attention_mask", "labels"],
    "speech-recognition": ["input_features", "labels"]
}

INVALID_DATASET_TYPE = "Dataset instance must be of type `Dataset`, got `{}`!"
INVALID_BATCH_TYPE = "A data batch must be a dictionary not `{}`!"
INVALID_DATASET_FIELDS = "Dataset samples must have the field `{}`!"


def create_dataloader(dataset, batch_size, shuffle, collate_fn):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


@pytest.mark.parametrize("task", DATASETS_MAPPING.keys())
def test_load_dataset(task):
    required_fields = TASK_TO_REQUIRED_FIELDS[task]

    path = DATASETS_MAPPING[task].pop("path")

    train_dataset = Dataset.load(
        path,
        split="train",
        **DATASETS_MAPPING[task]
    )
    assert isinstance(train_dataset, Dataset), INVALID_DATASET_TYPE.format(type(train_dataset))
    sample = train_dataset[0]
    assert isinstance(sample, Dict)
    for field in required_fields:
        assert field in sample, INVALID_DATASET_FIELDS.format(field)

    test_dataset = Dataset.load(
        path,
        split="test",
        **DATASETS_MAPPING[task]
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

    # Clean cache so that CI environment does not run out of space
    if CI_MODE == "TRUE":
        clean_cache(delay=1)