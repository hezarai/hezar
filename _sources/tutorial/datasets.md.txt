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

sentiment_dataset = Dataset.load("hezarai/sentiment-dksf", preprocessor="hezarai/bert-base-fa")  # A TextClassificationDataset instance
lscp_dataset = Dataset.load("hezarai/lscp-pos-500k", preprocessor="hezarai/roberta-base-fa")  # A SequenceLabelingDataset instance
xlsum_dataset = Dataset.load("hezarai/xlsum-fa", preprocessor="hezarai/t5-base-fa")  # A TextSummarizationDataset instance
...
```

```{note}
The `preprocessor` can also be a `Preprocessor` instance and does not have to be a path.
```python
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor

tokenizer = Preprocessor.load("hezarai/bert-base-fa")
dataset = Dataset.load("hezarai/sentiment-dksf", preprocessor=tokenizer)
```

The difference between using Hezar vs Hugging Face datasets is the output class. In Hezar when you load
a dataset using the `Dataset` class, it automatically finds the proper class for that dataset and creates a
PyTorch `Dataset` instance so that it can be easily passed to a PyTorch `DataLoader` class. That's why it also requires
the `preprocessor` to be filled since iterating the dataset needs the preprocessor to process the samples. 
```python
from torch.utils.data import DataLoader

from hezar.data.datasets import Dataset

dataset = Dataset.load("hezarai/lscp-pos-500k", preprocessor="hezarai/distilbert-base-fa")

loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.data_collator)
itr = iter(loader)
print(next(itr))
```
But when loading using Hugging Face datasets, the output is an HF Dataset instance.

```{note}
Any Hezar dataset can be loaded using HF datasets but not vise-versa!
(Because Hezar looks out for a `dataset_config.yaml` file in any dataset repo so non-Hezar datasets cannot be
loaded using Hezar `Dataset` class in most cases!)
```

## Dataset classes
Hezar categorizes datasets based on their target task. The dataset classes all inherit from the base `Dataset` class
which is a PyTorch Dataset subclass. (hence having `__getitem__` and `__len__` methods.)

Some examples of the dataset classes are `TextClassificationDataset`, `TextSummarizationDataset`, `SequenceLabelingDataset`, etc.

### Dataset Configs
All dataset classes must have a dataset config (as a dataclass of type `DatasetConfig`) implemented. Any field in such
dataclass is acceptable but the base dataset config takes the following for all configs:
- `task`: A mandatory field representing the task of the dataset of type `hezar.constants.TaskType`
- `path`: Path to the files of the dataset to be loaded. Can be a Hub dataset or local but remember that the loading procedure must be implemented in the `_load` method of the dataset.
- `max_size`: Maximum number of data samples. Overwrites the main length of the dataset when calling `len(dataset)`.
If set to a float value between 0 and 1, will be interpreted as a fraction value, e.g, 0.3 means 30% of the whole length.
- `hf_load_kwargs`: Keyword arguments to pass to the HF `datasets.load_dataset()`

### Custom Datasets
As mentioned, the `Dataset` class is a wrapper around the `torch.utils.data.Dataset` class to make it compatible with
Hezar's structure and standards (working with configurations most importantly). So writing your own custom dataset is
pretty easy.

The base class for all dataset classes in Hezar is the `hezar.data.Dataset` class which subclasses the `torch.utils.data.Dataset`
and adds its own functionalities to it.

Every dataset class must have the following signature:
```python
from dataclasses import dataclass

from hezar.data import Dataset, DatasetConfig
from hezar.registry import register_dataset


@dataclass
class ImageCaptioningDatasetConfig(DatasetConfig):
    name = "image_captioning"  # Must be unique among all datasets names
    attr1: int = ...
    attr2: str = ...
    attr3: bool = ...


@register_dataset("image_captioning", config_class=ImageCaptioningDatasetConfig)  # registering is optional
class ImageCaptioningDataset(Dataset):
    def __init__(self, config: ImageCaptioningDatasetConfig, split=None, preprocessor=None, **kwargs):
        super().__init__(config=config, split=split, preprocessor=preprocessor, **kwargs)
        # Custom initializations here
    
    def _load(self, split):
        """
        Load the data which can be a csv, HF dataset, etc. It will be assigned to the `self.data` attribute.
        """
        pass
    
    def __len__(self):
        """
        Typically returns the length of the `self.data` or the `config.max_size` if set.
        """
        pass
    
    def __getitem__(self, index):
        pass
```

## Loading Regular HF Datasets
All the current datasets provided in Hezar's Hugging Face, have the `dataset_config.yaml` in their repos which does not
exist for regular HF datasets. If you need to load such datasets (that have the correct structure and fields) in Hezar
using the `Dataset.load()` method, you have to provide the dataset config manually.

```{note}
If the dataset needs a configuration name to be specified (as `datasets.load_dataset(path, name=<configuration name>)`),
you can pass it either in `DatasetConfig.hf_load_kwargs` or like `Dataset.load("hezarai/dataset:config_name")`. See the
example below.
```

```python
from hezar.data import Dataset, SpeechRecognitionDatasetConfig

dataset_path = "mozilla-foundation/common_voice_17_0:fa"  # `fa` is the config name of the dataset

dataset_config = SpeechRecognitionDatasetConfig(
    path=dataset_path,
    labels_max_length=64,
)  # You can modify other fields too
dataset = Dataset.load(dataset_path, split="train", config=dataset_config)
```


## Dataset Templates (Deprecated)
We try to have a simple yet practical pattern for all datasets on the Hub. Every dataset on the Hub needs to have
a dataset loading script. Some ready to use templates are located in the [templates/dataset_scripts](https://github.com/hezarai/hezar/tree/main/templates/dataset_scripts) folder.
To add a new Hezar compatible dataset to the Hub you can follow the guide provided there.

```{note}
Dataset scripts are no longer recommended in the Hugging Face Hub but instead, the datasets must be uploaded in Parquet
format that already has all the required metadata inside them. For more info on how you can create or convert your
datasets to Parquet format see [this tutorial](https://huggingface.co/docs/datasets/process#export).
```

