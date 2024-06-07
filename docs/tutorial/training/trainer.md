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

## Trainer Class
The `Trainer` class is a full training/evaluation loop for models in Hezar (and with some tweaks, any PyTorch model). 
You only pass a couple of necessary objects (trainer config, model, datasets, preprocessor, etc.) to the Trainer and 
it takes care of the rest. Besides, you can also customize every way you can imagine! We've covered those low-level parts
in the guides too.

## Trainer Building Blocks
Let's explore what the Trainer is made of and how it works under the hood.

### TrainerConfig
The `TrainerConfig` is the main controlling part of the Trainer and all the training arguments are gathered by this class.

Let's explore all the accepted parameters:
- **task** (str, TaskType): The training task. Must be a valid name from `TaskType`.
- **output_dir** (str): Path to the directory to save trainer properties.
- **device** (str): Hardware device e.g, `cuda:0`, `cpu`, etc.
- **num_epochs** (int):Number of total epochs to train the model.
- **init_weights_from** (str): Path to a model from disk or Hub to load the initial weights from. Note that this only loads the model. weights and ignores other checkpoint-related states if the path is a checkpoint. To resume training from a checkpoint use the `resume_from_checkpoint` parameter.
- **resume_from_checkpoint** (bool, str, os.PathLike): Resume training from a checkpoint. If set to True, the trainer will load the latest checkpoint, otherwise if a path to a checkpoint is given, it will load that checkpoint and all the other states corresponding to that checkpoint.
- **max_steps** (int): Maximum number of iterations to train. This helps to limit how many batches you want to train in total.
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
- **gradient_accumulation_steps** (int): Number of updates steps to accumulate before performing a backward/update pass, defaults to 1.
- **distributed** (bool): Whether to use distributed training (via the `accelerate` package)
- **mixed_precision** (PrecisionType | str): Mixed precision type e.g, fp16, bf16, etc. (disabled by default)
- **use_cpu** (bool): Whether to train using the CPU only even if CUDA is available.
- **do_evaluate** (bool): Whether to run evaluation when calling `Trainer.train`
- **evaluate_with_generate** (bool): Whether to use `generate()` in the evaluation step or not. (only applicable for generative models).
- **metrics** (List[str | MetricConfig]): A list of metrics. Depending on the `valid_metrics` in the specific MetricsHandler of the Trainer. metric_for_best_model (str):Reference metric key to watch for the best model. Recommended to have a {train. | evaluation.} prefix (e.g, evaluation.f1, train.accuracy, etc.) but if not, defaults to `evaluation.{metric_for_best_model}`.
- **save_steps** (int): Save the trainer outputs every `save_steps` steps. Leave as `0` to ignore saving between training steps.
- **log_steps** (int): Save training metrics every `log_steps` steps.
- **checkpoints_dir** (str): Path to the checkpoints' folder. The actual files will be saved under `{output_dir}/{checkpoints_dir}`.
- **logs_dir** (str): Path to the logs' folder. The actual log files will be saved under `{output_dir}/{logs_dir}`.
