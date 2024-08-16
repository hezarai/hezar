# Sequence Labeling (POS, NER, etc.)

Sequence labeling (AKA token classification) is the task of assigning labels to individual tokens in a text e.g, 
part of speech tagging, named entity recognition, multi-word classification (chunker). In this tutorial we'll
walk through a training recipe for all those types of tasks.

We'll train a BERT model for POS tagging. 

Let's first import everything needed.
```python
from hezar.models import BertSequenceLabeling, BertSequenceLabelingConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig
```

As mentioned we'll use the base BERT model to do so.
```python
base_model_path = "hezarai/bert-base-fa"
```

## Dataset
We'll use a POS dataset called [LSCP](https://iasbs.ac.ir/~ansari/lscp/) which is a collection of millions of Persian tweets. We'll actually use a 
portion of this dataset which contains 500K samples from the dataset gathered by Hezar and hosted at 
[huggingface.co/datasets/hezarai/lscp-pos-500k](https://huggingface.co/datasets/hezarai/lscp-pos-500k).

Since the dataset provided there is ready to be used in Hezar out-of-the-box we'll stick to that but in case you want to
feed your own dataset it's pretty simple too. The Hezar's Trainer supports any iterable dataset like PyTorch's Dataset 
or a 🤗 Dataset's dataset. Indeed, you can subclass the Dataset class in Hezar too. For more info on how you can use 
custom datasets refer to [this tutorial]().

Loading Hezar datasets is pretty straight forward:

```python
train_dataset = Dataset.load("hezarai/lscp-pos-500k", split="train", preprocessor=base_model_path, max_size=100000)
eval_dataset = Dataset.load("hezarai/lscp-pos-500k", split="test", preprocessor=base_model_path, max_size=10000)
```
What are these objects? Well, these are basically PyTorch Dataset instances which are actually wrapped by Hezar's 
`SequenceLabelingDataset` class (a subclass of `hezar.data.datasets.Dataset`). 

What does the `SequenceLabelingDataset`'s `__getitem__` do? Simple! It just tokenizes the inputs and return a dictionary
containing `token_ids`, `labels`, `attention_mask`, `word_ids`.

Note that here we set the `max_size` for train and eval sets to 100K and 10K respectively since that's more than enough
for this demonstration.

### What about the data collator?
Indeed, our dataset needs a data collator function that concatenates multiple samples into a batch by padding them. etc.
Fortunately, Hezar supports all required data collator classes for all supported tasks (see `hezar.data.data_collators`)
and all Hezar dataset classes do have a `data_collator` attribute inside them by default but if you need to provide a
custom data collator you can pass it to the `Trainer` later.

## Model & Preprocessor
We will use the `BertSequenceLabeling` class from Hezar. Creating this model requires also it's config which itself needs
the `id2label` to be passed to it. We do have this `id2label` parameter in our dataset objects :) .

```python
model = BertSequenceLabeling(BertSequenceLabelingConfig(id2label=train_dataset.config.id2label))
```
Our model also needs a preprocessor (which is the tokenizer only). Since the BERT's tokenizer is the same for all tasks
we'll load it from the `base_model_path` defined earlier above.
```python
preprocessor = Preprocessor.load(base_model_path)
```

## Training Configuration
Every training requires a configuration using the `TrainerConfig` class. For this task we just need to fill in some 
casual parameters as below:
```python
train_config = TrainerConfig(
    output_dir="bert-fa-pos-lscp-500k",
    task="sequence_labeling",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_dataloader_workers=2,
    num_epochs=5,
    metrics=["seqeval"],
)
```
Notes:
- Since our model has no pretrained weights, we set the `init_weights_from` to load pretrained weights from a path (local or Hub).
- To evaluate the model while training, we need a proper metric module hence setting `seqeval` which is suitable for all token classification tasks.


Now it's time to create the Trainer object. 
```python
trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
```

Now we've got everything needed to train the model on our dataset. Let's roll...
```python
trainer.train()
```
```
Hezar (WARNING): Partially loading the weights as the model architecture and the given state dict are incompatible! 
Ignore this warning in case you plan on fine-tuning this model
Incompatible keys: []
Missing keys: ['classifier.weight', 'classifier.bias']


******************** Training Info ********************

  Output Directory: bert-fa-pos-lscp-500k
  Task: sequence_labeling
  Model: BertSequenceLabeling
  Init Weights: hezarai/bert-base-fa
  Device(s): cuda
  Batch Size: 8
  Epochs: 5
  Total Steps: 62500
  Training Dataset: SequenceLabelingDataset(100000)
  Evaluation Dataset: SequenceLabelingDataset(10000)
  Optimizer: adam
  Scheduler: None
  Initial Learning Rate: 2e-05
  Learning Rate Decay: 0.0
  Number of Parameters: 118315031
  Number of Trainable Parameters: 118315031
  Mixed Precision: Full (fp32)
  Gradient Accumulation Steps: 1
  Metrics: ['seqeval']
  Save Steps: 12500
  Log Steps: None
  Checkpoints Path: bert-fa-pos-lscp-500k/checkpoints
  Logs Path: bert-fa-pos-lscp-500k/logs/Jun14_00-46-47_bigrig

*******************************************************


Epoch: 1/5      100%|######################################################################| 12500/12500 [14:29<00:00, 14.37batch/s, loss=0.358]
Evaluating...   100%|######################################################################| 1250/1250 [00:25<00:00, 56.82batch/s, accuracy=0.914, f1=0.914, loss=0.254, precision=0.914, recall=0.914]

Epoch: 2/5      100%|######################################################################| 12500/12500 [14:19<00:00, 14.17batch/s, loss=0.358]
Evaluating...   100%|######################################################################| 1250/1250 [00:23<00:00, 56.82batch/s, accuracy=0.926, f1=0.926, loss=0.254, precision=0.926, recall=0.926]

Epoch: 3/5      100%|######################################################################| 12500/12500 [14:39<00:00, 14.02batch/s, loss=0.358]
Evaluating...   100%|######################################################################| 1250/1250 [00:22<00:00, 56.82batch/s, accuracy=0.928, f1=0.928, loss=0.254, precision=0.928, recall=0.928]

Epoch: 4/5      100%|######################################################################| 12500/12500 [14:22<00:00, 14.56batch/s, loss=0.358]
Evaluating...   100%|######################################################################| 1250/1250 [00:20<00:00, 56.82batch/s, accuracy=0.932, f1=0.932, loss=0.254, precision=0.932, recall=0.932]

Epoch: 5/5      100%|######################################################################| 12500/12500 [14:29<00:00, 14.98batch/s, loss=0.358]
Evaluating...   100%|######################################################################| 1250/1250 [00:20<00:00, 56.82batch/s, accuracy=0.942, f1=0.942, loss=0.254, precision=0.942, recall=0.942]
Hezar (INFO): Training done!
```
## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome POS model!")
```
