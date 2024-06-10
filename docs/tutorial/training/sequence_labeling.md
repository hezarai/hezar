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
train_dataset = Dataset.load("hezarai/lscp-pos-500k", split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load("hezarai/lscp-pos-500k", split="test", tokenizer_path=base_model_path)
```
What are these objects? Well, these are basically PyTorch Dataset instances which are actually wrapped by Hezar's 
`SequenceLabelingDataset` class (a subclass of `hezar.data.datasets.Dataset`). 

What does the `SequenceLabelingDataset`'s `__getitem__` do? Simple! It just tokenizes the inputs and return a dictionary
containing `token_ids`, `labels`, `attention_mask`, `word_ids`. 

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

## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome POS model!")
```
