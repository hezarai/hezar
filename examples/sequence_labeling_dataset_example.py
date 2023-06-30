from torch.utils.data import DataLoader

from hezar.data.datasets import SequenceLabelingDataset, SequenceLabelingDatasetConfig


config = SequenceLabelingDatasetConfig(
    name="sequence_labeling",
    path="hezarai/lscp-500k",
    tokens_field="tokens",
    tags_field="pos_tags",
    tokenizer_path="hezarai/distilbert-base-fa",
)

dataset = SequenceLabelingDataset(config, split="train")
print(dataset[0])

loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.data_collator)
itr = iter(loader)
x = next(itr)
print(x)
