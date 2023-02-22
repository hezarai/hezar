from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.data import DataCollatorWithPadding

from ...configs import DatasetConfig
from ...preprocessors.tokenizer import Tokenizer
from ...registry import register_dataset


@dataclass
class TextClassificationDatasetConfig(DatasetConfig):
    name = "text_classification"
    task = "text_classification"
    path: str = None
    preprocessors: List[str] = None
    tokenizer_path: str = None
    label_field: str = None
    text_field: str = None


@register_dataset("text_classification", config_class=TextClassificationDatasetConfig)
class TextClassificationDataset(Dataset):
    def __init__(self, config: TextClassificationDatasetConfig, split=None, **kwargs):
        self.config = config.update(kwargs)
        self.dataset = self._load(split)
        self._extract_labels()
        self.preprocessor = Tokenizer.load(self.config.tokenizer_path)
        self.data_collator = DataCollatorWithPadding(self.preprocessor.tokenizer)

    def _load(self, split):
        dataset = load_dataset(self.config.path, split=split)
        return dataset

    def _extract_labels(self):
        labels_list = self.dataset.to_pandas()[self.config.label_field].unique().tolist()
        self.id2label = self.config.id2label = {str(k): str(v) for k, v in dict(list(enumerate(labels_list))).items()}
        self.label2id = self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(labels_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index][self.config.text_field]
        label = self.dataset[index][self.config.label_field]
        inputs = self.preprocessor(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        label_idx = int(self.label2id[str(label)])
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["label"] = label_idx

        return inputs
