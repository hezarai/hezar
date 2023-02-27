from torch.utils.data import DataLoader

from hezar.data.datasets.text_classification import TextClassificationDataset, TextClassificationDatasetConfig

config = TextClassificationDatasetConfig(
    name="text_classfication",
    path="mteb/amazon_massive_intent",
    text_field="text",
    label_field="label_text",
    tokenizer_path="hezar-ai/distilbert-fa",
)

dataset = TextClassificationDataset(config, split="train")
print(dataset[0])

loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.data_collator)
itr = iter(loader)
x = next(itr)
print(x)
