from hezar.data.datasets.text_classification import TextClassificationDataset, TextClassificationDatasetConfig

config = TextClassificationDatasetConfig(
    name="text_classfication",
    path="farsi_news",
    text_field="title",
    label_field="tags",
    tokenizer_path="hezar-ai/distilbert-fa"
)

dataset = TextClassificationDataset(config, split='hamshahri')
print(dataset)
