# Training & Fine-tuning

Training a model in Hezar is pretty much like any other library or even simpler! As mentioned before, any model in Hezar
is also a PyTorch module. So training a model is actually training a PyTorch model with some more cool features!
Let's dive in.

## Setup
In this example we're going to train a sentiment analysis model based on DistilBERT on a dataset containing
text and sentiment pairs collected from SnappFood/Digikala user comments.
### Import everything needed
First things first, let's import the required stuff.

```python
from hezar.models import DistilBertTextClassification, DistilBertTextClassificationConfig
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig
from hezar.preprocessors import Preprocessor
```
### Define paths
Let's define our paths to the datasets, tokenizer, etc.
```python
DATASET_PATH = "hezarai/sentiment-dksf"  # dataset path on the Hub
BASE_MODEL_PATH = "hezarai/distilbert-base-fa"  # used as model backbone weights and tokenizer
```
## Datasets
We can easily load our desired datasets from the Hub.
```python
train_dataset = Dataset.load(DATASET_PATH, split="train", tokenizer_path=BASE_MODEL_PATH)
eval_dataset = Dataset.load(DATASET_PATH, split="test", tokenizer_path=BASE_MODEL_PATH)
```

## Model
Let's build our model along with its tokenizer.
### Build the model
```python
model = DistilBertTextClassification(DistilBertTextClassificationConfig(id2label=train_dataset.config.id2label))
```
### Load the tokenizer
The tokenizer can be loaded from the base model path.
```python
tokenizer = Preprocessor.load(BASE_MODEL_PATH)
```

## Trainer
Hezar has a general Trainer class that satisfies most of your needs. You can customize almost every single part of it
but for now, we stick with the base class `Trainer`.
### Trainer Config
Define all the training properties in the trainer's config. As we're training a text classification model we set the
task to `text_classification` in our config. Other parameters are also customizable like below:
```python
train_config = TrainerConfig(
    output_dir="distilbert-fa-sentiment-analysis-dksf",
    task="text_classification",
    device="cuda",
    init_weights_from=BASE_MODEL_PATH,
    batch_size=8,
    num_epochs=5,
    metrics=["f1"],
    num_dataloader_workers=0,
    seed=42,
    optimizer="adamw",
    learning_rate=2e-5,
    weight_decay=.0,
    scheduler="reduce_on_plateau",
    use_amp=False,
    save_freq=1,
)
```
### Setup the Trainer
Now that we have our training config we can setup the Trainer.
```python
trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=tokenizer,
)
```
### Start Training
```python
trainer.train()
```
```
Epoch: 1/5      100%|####################################| 3576/3576 [07:07<00:00,  8.37batch/s, f1=0.732, loss=0.619]
Evaluating...   100%|####################################| 290/290 [00:07<00:00, 38.64batch/s, f1=0.8, loss=0.473]
Epoch: 2/5      100%|####################################| 3576/3576 [07:00<00:00,  8.50batch/s, f1=0.807, loss=0.47]
Evaluating...   100%|####################################| 290/290 [00:07<00:00, 39.87batch/s, f1=0.838, loss=0.419]
Epoch: 3/5      100%|####################################| 3576/3576 [07:01<00:00,  8.48batch/s, f1=0.864, loss=0.348]
Evaluating...   100%|####################################| 290/290 [00:07<00:00, 39.97batch/s, f1=0.875, loss=0.346]
Epoch: 4/5      100%|####################################| 3576/3576 [06:57<00:00,  8.56batch/s, f1=0.919, loss=0.227]
Evaluating...   100%|####################################| 290/290 [00:07<00:00, 38.84batch/s, f1=0.875, loss=0.381]
Epoch: 5/5      100%|####################################| 3576/3576 [07:02<00:00,  8.46batch/s, f1=0.943, loss=0.156]
Evaluating...   100%|####################################| 290/290 [00:07<00:00, 39.71batch/s, f1=0.887, loss=0.446]
```
### Evaluate
```python
trainer.evaluate()
```
```
Evaluating...   100%|####################################| 290/290 [00:07<00:00, 39.46batch/s, f1=0.887, loss=0.445]
```
## Push everything
Now you can push your trained model to the Hub. The files to push are the model, model config, preprocessor, trainer config,
etc.
```python
trainer.push_to_hub("arxyzan/distilbert-fa-sentiment-dksf")
```
