# Datasets
Hezar provides both dataset class implementations and ready-to-use data files for the community.

## Hub Datasets
Hezar datasets are all hosted on the Hugging Face Hub and can be loaded just like any dataset on the Hub.

### Load using Hugging Face datasets
```python
from datasets import load_dataset

sentiment_dataset = load_dataset("hezarai/sentiment-dksf")
lscp_dataset = load_dataset("hezarai/lscp-pos-500k")
xlsum_dataset = load_dataset("hezarai/xlsum-fa")
...
```

### Load using Hezar Dataset
```python
from hezar.data import Dataset

sentiment_dataset = Dataset.load("hezarai/sentiment-dksf")  # A TextClassificationDataset instance
lscp_dataset = Dataset.load("hezarai/lscp-pos-500k")  # A SequenceLabelingDataset instance
xlsum_dataset = Dataset.load("hezarai/xlsum-fa")  # A TextSummarizationDataset instance
...
```

The difference between using Hezar vs Hugging Face datasets is the output class. In Hezar when you load
a dataset using the `Dataset` class, it automatically finds the proper class for that dataset and creates a
PyTorch `Dataset` instance so that it can be easily passed to a PyTorch `DataLoader` class.
```python
from torch.utils.data import DataLoader

from hezar.data.datasets import Dataset

dataset = Dataset.load(
    "hezarai/lscp-pos-500k",
    tokenizer_path="hezarai/distilbert-base-fa",  # tokenizer_path is necessary for data collator
)

loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.data_collator)
itr = iter(loader)
print(next(itr))
```
But when loading using Hugging Face datasets, the output is an HF Dataset instance.

So in a nutshell, any Hezar dataset can be loaded using HF datasets but not vise-versa!
(Because Hezar looks out for a `dataset_config.yaml` file in any dataset repo so non-Hezar datasets cannot be
loaded using Hezar `Dataset` class.)

## Dataset classes
Hezar categorizes datasets based on their target task. The dataset classes all inherit from the base `Dataset` class
which is a PyTorch Dataset subclass. (hence having `__getitem__` and `__len__` methods.)

Some examples of the dataset classes are `TextClassificationDataset`, `TextSummarizationDataset`, `SequenceLabelingDataset`, etc.

## Dataset Templates
We try to have a simple yet practical pattern for all datasets on the Hub. Every dataset on the Hub needs to have
a dataset loading script. Some ready to use templates are located in the [templates/dataset_scripts](https://github.com/hezarai/hezar/tree/main/templates/dataset_scripts) folder.
To add a new Hezar compatible dataset to the Hub you can follow the guide provided there.
