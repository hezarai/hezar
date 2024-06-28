# Text Classification (Sentiment Analysis)
Text classification is the task of categorizing a text to a label. The most notable example is sentiment analysis.
In this tutorial we'll finetune a BERT model on a dataset of sentiments (labels: positive, negative, neutral) gathered
from users comments on Digikala and SnappFood.

Let's first import everything needed.
```python
from hezar.models import BertTextClassification, BertTextClassificationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig
```

As mentioned we'll use the base BERT model to do so.
```python
base_model_path = "hezarai/bert-base-fa"
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
    
    def __init__(self, config: TextClassificationDatasetConfig, split=None, preprocessor=None, **kwargs):
        super().__init__(config, split=split, preprocessor=preprocessor, **kwargs)

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
model = BertTextClassification(BertTextClassificationConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)
```

## Training
Now that the datasets and model are ready let's go for training.
### Training Configuration
```python
train_config = TrainerConfig(
    output_dir="bert-fa-sentiment-analysis",
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
```
Hezar (WARNING): Partially loading the weights as the model architecture and the given state dict are incompatible! 
Ignore this warning in case you plan on fine-tuning this model
Incompatible keys: []
Missing keys: ['classifier.weight', 'classifier.bias']


******************** Training Info ********************

  Output Directory: bert-fa-sentiment-analysis-dksf
  Task: text_classification
  Model: BertTextClassification
  Init Weights: hezarai/bert-base-fa
  Device(s): cuda
  Batch Size: 8
  Epochs: 10
  Total Steps: 35760
  Training Dataset: TextClassificationDataset(28602)
  Evaluation Dataset: TextClassificationDataset(2315)
  Optimizer: adam
  Scheduler: None
  Initial Learning Rate: 2e-05
  Learning Rate Decay: 0.0
  Number of Parameters: 118299651
  Number of Trainable Parameters: 118299651
  Mixed Precision: Full (fp32)
  Gradient Accumulation Steps: 1
  Metrics: ['accuracy']
  Save Steps: 3576
  Log Steps: None
  Checkpoints Path: bert-fa-sentiment-analysis-dksf/checkpoints
  Logs Path: bert-fa-sentiment-analysis-dksf/logs/Jun13_19-29-38_bigrig

*******************************************************


Epoch: 1/10     100%|######################################################################| 3576/3576 [05:58<00:00,  9.99batch/s, loss=0.62] 
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 38.38batch/s, accuracy=0.811]

Epoch: 2/10     100%|######################################################################| 3576/3576 [06:02<00:00,  9.86batch/s, loss=0.479]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 38.66batch/s, accuracy=0.818]

Epoch: 3/10     100%|######################################################################| 3576/3576 [06:03<00:00,  9.84batch/s, loss=0.369]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 37.79batch/s, accuracy=0.845]  

Epoch: 4/10     100%|######################################################################| 3576/3576 [06:00<00:00,  9.92batch/s, loss=0.299]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 38.09batch/s, accuracy=0.829]

Epoch: 5/10     100%|######################################################################| 3576/3576 [06:00<00:00,  9.91batch/s, loss=0.252]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 37.88batch/s, accuracy=0.854]

Epoch: 6/10     100%|######################################################################| 3576/3576 [06:02<00:00,  9.87batch/s, loss=0.219]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 38.68batch/s, accuracy=0.846] 

Epoch: 7/10     100%|######################################################################| 3576/3576 [06:00<00:00,  9.93batch/s, loss=0.194]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 37.61batch/s, accuracy=0.862] 

Epoch: 8/10     100%|######################################################################| 3576/3576 [06:01<00:00,  9.90batch/s, loss=0.175]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 37.85batch/s, accuracy=0.857] 

Epoch: 9/10     100%|######################################################################| 3576/3576 [06:01<00:00,  9.90batch/s, loss=0.16] 
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 37.71batch/s, accuracy=0.84] 

Epoch: 10/10    100%|######################################################################| 3576/3576 [06:01<00:00,  9.89batch/s, loss=0.147]
Evaluating...   100%|######################################################################| 290/290 [00:07<00:00, 37.70batch/s, accuracy=0.83] 
Hezar (INFO): Training done!
```
## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome sentiment analysis model!")
```


