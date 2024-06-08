# Trainer Basics

Before exploring training recipes for different tasks, lets dive into the Trainer itself! The `Trainer` is the base
super class that handles everything needed in order to train any model in Hezar on a given dataset. So lets begin.

## A Simple Example

The Trainer abstracts all the unnecessary details so that your training script is as minimal as possible. Below is an
example to train a CRNN model for license plate recognition.

```python
from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import Preprocessor
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig

base_model_path = "hezarai/crnn-base-fa-64x256"

train_dataset = Dataset.load("hezarai/persian-license-plate-v1", split="train", max_length=8, reverse_digits=True)
eval_dataset = Dataset.load("hezarai/persian-license-plate-v1", split="test", max_length=8, reverse_digits=True)

model = CRNNImage2Text(CRNNImage2TextConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    output_dir="crnn-plate-fa-v1",
    task="image2text",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=20,
    metrics=["cer"],
    metric_for_best_model="cer"
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
trainer.train()
```

## Trainer API

The `Trainer` class is a full training/evaluation loop for models in Hezar (and with some tweaks, any PyTorch model).
You only pass a couple of necessary objects (trainer config, model, datasets, preprocessor, etc.) to the Trainer and
it takes care of the rest. Besides, you can also customize every way you can imagine! We've covered those low-level
parts
in the guides too.

## Trainer Building Blocks

Let's explore what the Trainer is made of and how it works under the hood.

### TrainerConfig

The `TrainerConfig` is the main controlling part of the Trainer and all the training arguments are gathered by this
class.

Let's explore all the available parameters:

- **task** (str, TaskType): The training task. Must be a valid name from `TaskType`.
- **output_dir** (str): Path to the directory to save trainer properties.
- **device** (str): Hardware device e.g, `cuda:0`, `cpu`, etc.
- **num_epochs** (int):Number of total epochs to train the model.
- **init_weights_from** (str): Path to a model from disk or Hub to load the initial weights from. Note that this only
  loads the model. weights and ignores other checkpoint-related states if the path is a checkpoint. To resume training
  from a checkpoint use the `resume_from_checkpoint` parameter.
- **resume_from_checkpoint** (bool, str, os.PathLike): Resume training from a checkpoint. If set to True, the trainer
  will load the latest checkpoint, otherwise if a path to a checkpoint is given, it will load that checkpoint and all
  the other states corresponding to that checkpoint.
- **max_steps** (int): Maximum number of iterations to train. This helps to limit how many batches you want to train in
  total.
- **num_dataloader_workers** (int): Number of dataloader workers, defaults to 4 .
- **dataloader_shuffle** (bool): Control dataloaders `shuffle` argument.
- **seed** (int): Control determinism of the run by setting a seed value. defaults to 42.
- **optimizer** (OptimizerType): Name of the optimizer, available values include properties in `OptimizerType` enum.
- **learning_rate** (float): Initial learning rate for the optimizer.
- **weight_decay** (float): Optimizer weight decay value.
- **lr_scheduler** (LRSchedulerType): Optional learning rate scheduler among `LRSchedulerType` enum.
- **lr_scheduler_kwargs** (Dict[str, Any]): LR scheduler instructor kwargs depending on the scheduler type
- **batch_size** (int): Training batch size.
- **eval_batch_size** (int): Evaluation batch size, defaults to `batch_size` if None.
- **gradient_accumulation_steps** (int): Number of updates steps to accumulate before performing a backward/update pass,
  defaults to 1.
- **distributed** (bool): Whether to use distributed training (via the `accelerate` package)
- **mixed_precision** (PrecisionType | str): Mixed precision type e.g, fp16, bf16, etc. (disabled by default)
- **use_cpu** (bool): Whether to train using the CPU only even if CUDA is available.
- **do_evaluate** (bool): Whether to run evaluation when calling `Trainer.train`
- **evaluate_with_generate** (bool): Whether to use `generate()` in the evaluation step or not. (only applicable for
  generative models).
- **metrics** (List[str | MetricConfig]): A list of metrics. Depending on the `valid_metrics` in the specific
  MetricsHandler of the Trainer. metric_for_best_model (str):Reference metric key to watch for the best model.
  Recommended to have a {train. | evaluation.} prefix (e.g, evaluation.f1, train.accuracy, etc.) but if not, defaults
  to `evaluation.{metric_for_best_model}`.
- **save_steps** (int): Save the trainer outputs every `save_steps` steps. Set to `0` or `None` to ignore saving between
  training steps.
- **log_steps** (int): Save training metrics every `log_steps` steps.
- **checkpoints_dir** (str): Path to the checkpoints' folder. The actual files will be saved
  under `{output_dir}/{checkpoints_dir}`.
- **logs_dir** (str): Path to the logs' folder. The actual log files will be saved under `{output_dir}/{logs_dir}`.

### Datasets

The datasets must be explicitly passed to the Trainer when instantiating it:

```python
Trainer(
    ...
train_dataset = ...,
eval_dataset = ...,
...
)
```

The datasets passed to the Trainer can be any PyTorch-like dataset class. It's recommended to pass
a `hezar.data.Dataset`
instance so that all special features are enabled. To find out how you can define your own dataset by subclassing
a `Dataset` class in Hezar, refer to the [datasets tutorial](../datasets.md).

```{note}
In general, any class that has the `__len__` and `__getitem__` methods implemented is supported. (e.g, PyTorch's `Dataset` class)
```

### Model

The model passed to the Trainer can be any `hezar.models.Model` instance. You can easily create your own custom models
by subclassing this base class. For more info, refer to the [models tutorial](../models.md).

You won't need to map your model to a device since that's handled by the Trainer (by setting the `device` parameter in
the `TrainerConfig`).

```{note}
All device placements regarding model, data, etc. is handled internally by the Trainer and there
is no need to manually do so beforehand. Just specify the `device` parameter and the rest is taken
care of.
```

How about pretrained weights? well, if you want to finetune the same model on new data, the simplest way is to just load
it beforehand and then pass it to the Trainer like below:

```
model = Model.load(pretrained_path)

trainer = Trainer(
    model=model,
    ...
)
```

But, if you want to finetune a model for a different task but load the pretrained weights onto it (e.g, Loading the
pretrained weights of `bert-base` on a `BertTextClassification` that has an additional classification head) you can
specify the `init_weights_from` parameter in the Trainer's config.

```python
base_model_path = "hezarai/bert-base-fa"
model = BertTextClassification(BertTextClassificationConfig(id2label=id2label))  # Randomly initialized model

trainer_config = TrainerConfig(
    ...,
    init_weights_from=base_model_path,
    ...
)
trainer = Trainer(
    trainer_config,
    ...
)
```

### Optional Parameters

The Trainer's instantiation requires the trainer config, model and training dataset. Now lets explore optional inputs
to the Trainer's `__init__` function.

- `eval_dataset`: The evaluation dataset to evaluate the model performance on it.
- `data_collator`: Custom data collator or collate function. This will default to the `data_collator` attribute of the
  dataset.
- `preprocessor`: If the model's `preprocessor` is None, you must pass the `preprocessor` to the Trainer.
- `metrics_handler`: A custom `MetricsHandler` instance responsible for custom evaluation metrics implementations.
- `optimizer`: A custom optimizer (`torch.optim.Optimizer`). Pass custom optimizer to this parameter since casual ones (
  Adam, AdamW, SGD) are supported by the Trainer and can be set in `TrainerConfig.optimizer`.
- `lr_scheduler`: Same as the optimizer. Can be also set by name in `TrainerConfig.lr_scheduler`.
- `accelerator`: A custom `Accelerator` instance to handle model and data sharding for distributed environments the
  Trainer. Uses the 🤗 Accelerate package which also takes care of mixed precision, gradient accumulation, distributed
  logging, etc.

### What Actually Happens in Trainer's `__init__`?

When the Trainer instantiates the following properties are created:

- Model's weight initialization
- Accelerator
- Optimizer and LR Scheduler (If provided)
- Metrics handler
- All info regarding the training steps and criterias
- Loading checkpoint (if availabel and set in the config)
- Logging modules (Tensorboard and CSV writer)
- Trainer state (A module for tracking the state of the trainer on different events)
- Global seeding
- Training loss tracker

## Training Loop

After initializing the Trainer, the only step left is to call the `.train()` method. But a lot of magic happens inside
so that a model gets trained. Let's explore them.

### Inner Training Loop
The inner training loop (`Trainer.inner_training_loop(epoch_num)`) trains the model on the whole train
dataset.

**1. Create the train data loader**

The dataloader is created from the train dataset but not necessarily the whole data
but depending on the `max_steps` or resuming from checkpoint the data loader might be different among different epochs.
(The sampler's indices order is always the same even if shuffle is enabled among different runs and calls).
The method responsible for this is `create_train_dataloader(dataset)` and can be overridden in custom Trainers.

**2. Start Training Iteration**

The inner training loop to train the model on the whole data loader batches starts and repeats the following loop for
the number of batches present in the data loader:

1. Prepare input batch (`prepare_input_batch()`): Responsible for casting and other sanitizations on the data batches if
   needed. The default behavior does absolutely nothing and the main usage of this method is for custom Trainers.
2. Training step (`training_step()`): Does the forward/backward operation and returns the loss. Note that optimizer step
   does not happen here since it has its own method.
3. Optimization step (`optimization_step`): Does the optimizer stepping and zeros gradients afterward. (Gradient
   accumulation is handled by the accelerator)
4. Update loss tracker and the trainer states.
5. Update and show the loss moving average in the progress bar.
6. Perform saving and logging according to `save_steps` and `log_steps`.
7. Return average loss up until now. (This value is accumulated and averaged since the beginning of the whole training
   process and does not demonstrate the mean loss value throughout individual epochs.)

### Evaluation Loop

The evaluation loop (`Trainer.evaluate()`) evaluates the current model on the evaluation dataset.

```{note}
The `evaluate()` method can also be called independently. It can also be called to evaluate the model on a custom dataset (`trainer.evaluate(eval_dataset)`). But note that no logging happens when calling this method like that!
```

The evaluation loop simply runs prediction on the samples and the metrics handler computes the metric results (Based
on `TrainerConfig.task`). Each task has its own metrics handler class which you can find
at `hezar.trainer.metrics_handlers` that leverage the `hezar.metrics` modules.

### Post Training/Evaluation Steps

After one training loop on the `train_dataset` and a full evaluation on the `eval_dataset`, the results
are gathered by the trackers and logged to logging modules (Tensorboard, CSV, etc.) and checkpoints are saved.

## Additional Operations

### Resuming Training

By default, the Trainer will only save checkpoints at the end of each epoch, but you can change this behavior by setting
the `save_steps` argument in the Trainer's config. Note that one step means one step means a single call to
the `training_step()`. The total number of steps in one epoch equals `len(train_dataset)`/`batch_size` and the total
steps equals `len(train_dataset)`/`batch_size`*`num_train_epochs`.

To enable resuming from checkpoint, you just need to set `resume_from_checkpoint=True` in the Trainer's config:
```python
trainer_config = TrainerConfig(
    ...,
    resume_from_checkpoint=True,
    ...
)
```
For large models or datasets, it's recommended to also set the `save_steps` argument in the Trainer's config:
```python
trainer_config = TrainerConfig(
    ...,
    resume_from_checkpoint=True,
    save_steps=1000,
    ...
)
```
```{note}
Checkpoints will be saved at the end of each epoch anyway so there's no need to set the `save_steps` to something dividable by the length of the data loader.
```

```{note}
If a checkpoint is set to a step in the middle of the epoch, the Trainer will resume from that point in the data loader
using a ranged sampler that firstly, handles creating the same indices with the same order even on shuffle mode and
secondly, the correct batch indice to continue from.
```

You can also set the `resume_from_checkpoint` to a path to a checkpoint. By doing so, model and optimizer state dicts will be loaded from that path and the trainer state will figure out other parameters based on the name of the checkpoint
if possible (since checkpoint names are based on the number of the `global_step`).

### Distributed Training
Hezar's Trainer is compatible with 🤗 Accelerate. You just set the `distributed` argument to `True` in the Trainer's config
and the rest is handled by the 🤗 Accelerate package by running the `accelerate launch` command. 
Refer to [🤗 Accelerate docs](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch)
for more info.

### Mixed Precision
Mixed precision can also be acheived by setting the `mixed_precision` argument in the Trainer's config.
```python
trainer_config = TrainerConfig(
    ...,
    mixed_precision="bf16",  # Also accepts `fp16` and `full`
    ...
)
```

### Gradient Accumulation
Gradient accumulation is a technique for training on larger batch sizes without increasing the batch size directly. If
the `gradient_accumulation_steps` is set to T and batch size is N, it simulates training with a batch size of N*T by 
accumulating the losses for every T steps and averaging over them and then performing the backward operation.

```{note}
Setting the `gradient_accumulation_steps` to 1 (default behavior) is exactly what happens in the regular training without
any accumulation.
```

## Custom Trainer
To implement your own custom Trainer, refer to [this tutorial](../../guide/advanced_training.md).
