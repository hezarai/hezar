# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader

from hezar.data.datasets import Dataset

dataset = Dataset.load("hezarai/lscp-pos-500k", tokenizer_path="hezarai/distilbert-base-fa")

loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.data_collator)
itr = iter(loader)
print(next(itr))
