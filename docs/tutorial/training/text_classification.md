# Text Classification (Sentiment Analysis)
Text classification is the task of categorizing a text to a label. The most notable example is sentiment analysis.
In this tutorial we'll finetune a RoBERTa model on a dataset of sentiments (labels: positive, negative, neutral) gathered
from users comments on Digikala and SnappFood.

Let's first import everything needed.
```python
from hezar.models import RobertaTextClassification, RobertaTextClassificationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig
```

As mentioned we'll use the base RoBERTa model to do so.
```python
base_model_path = "hezarai/roberta-base-fa"
```
## Dataset
The selected dataset is a collection of users comments on products and food.

Since the dataset provided there is ready to be used in Hezar out-of-the-box we'll stick to that but in case you want to
feed your own dataset it's pretty simple too. The Hezar's Trainer supports any iterable dataset like PyTorch's Dataset 
or a 🤗 Dataset's dataset. Indeed, you can subclass the Dataset class in Hezar too. 

### Option 1: Hezar Sentiment Dataset
Loading Hezar datasets is pretty straight forward:
```python
train_dataset = Dataset.load(dataset_path, split="train", preprocessor=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", preprocessor=base_model_path)
```

### Option 2: Custom Sentiment Dataset
Let's see how to create a custom dataset for text classification. When it comes to customizing a dataset with a supported task in Hezar,
there are two ways in general; Subclassing the dataset class of that task in particular and subclassing the base `Dataset`
class. 

Since we're customizing an `text_classification` dataset, we can override the `TextClassificationDataset` class.

Let's consider you have a CSV file of your dataset with two columns: `text`, `label`.

```python
import pandas as pd

from hezar.data import TextClassificationDataset, TextClassificationDatasetConfig

class SentimentAnalysisDataset(TextClassificationDataset):
    id2label = {0: "negative", 1: "positive", 2: "neutral"}
    
    def __init__(self, config: TextClassificationDatasetConfig, split=None, **kwargs):
        super().__init__(config, split=split, **kwargs)

    def _load(self, split):
        # Load a dataframe here and make sure the split is fetched
        df = pd.read_csv(self.config.path)
        return df


    def _extract_labels(self):
        """
        Extract label names, ids and build dictionaries.
        """
        self.label2id = self.config.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = self.config.num_labels = len(self.id2label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][self.config.text_field]
        label = self.data[index][self.config.label_field]
        label_id = self.label2id[label]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation_strategy="longest_first",
            padding="longest",
            return_attention_mask=True,
        )
        label_idx = torch.tensor([label_id], dtype=torch.long)  # noqa
        inputs["labels"] = label_idx

        return inputs
```

### Data Collator
The data collator of such dataset does nothing but padding the values of the input batch fields like `token_ids`, `attention_mask`, etc.

The default `data_collator` of the `TextClassificationDataset` is `TextPaddingDataCollator` and if that's okay for your
dataset just move on to the next steps.

## Model & Preprocessor
As said, we'll use the base RoBERTa model:
```python
model = RobertaTextClassification(RobertaTextClassificationConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)
```

## Training
Now that the datasets and model are ready let's go for training.
### Training Configuration
```python
train_config = TrainerConfig(
    output_dir="roberta-fa-sentiment-analysis",
    task="text_classification",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
    metrics=["f1", "precision", "accuracy", "recall"],
)
```
- Setting `init_weights_from=base_model_path` loads the model weights from `base_model_path`. Note that this will warn
that some weights are not compatible or missing which is fine since the base model does not have the final classifier layer.
- For this classification task we choose to use [`f1`, `accuracty`, `precision`, `recall`]
- Other training arguments can also be modified or set. Refer to the [Trainer's tutorial](trainer.md)


### Create the Trainer
```python
trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    preprocessor=preprocessor,
)
trainer.train()
```

## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome sentiment analysis model!")
```


