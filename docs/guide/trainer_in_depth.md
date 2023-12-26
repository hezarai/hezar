# Trainer In-depth Guide
The `Trainer` is the base class for training all the models in Hezar no matter the task, dataset, model architecture, etc.
In this guide, we'll demonstrate all the internals and details in this powerful class and how you can customize it based
on your needs.

We'll do this step by step in the exact order that takes place when training a model.

## Initialization
In order to initialize the Trainer you would need to have some objects ready.

- **Trainer Config**: A Trainer config is a config dataclass (of type `TrainerConfig`) with a bunch of attributes that configures
how the Trainer behaves. The most important ones being:
  - `output_dir` (required): Path to the directory to save trainer outputs (checkpoints, logs, etc.)
  - `task` (required): Specify the task of the training e.g, `text_classification`, `speech_recognition`, etc.
  - `num_epochs` (required): Number of training epochs
  - `init_weights_from` (optional): If the model is randomly initialized or you want to finetune it from a different set of
  weights, you can provide a path to a pretrained model using this parameter to load the model weights from.
  - `batch_size` (required): Training batch size
  - `eval_batch_size` (optional): Evaluation batch size (defaults to `batch_size` if not set)
  - `mixed_precision` (optional): Type of mixed precision e.g, `fp16`, `bf16`, etc. (Disabled by default)
  - `metrics` (optional): A set of metrics for model evaluation. Available metrics can be obtained using `utils.list_available_metrics()`
  - and etc.
- **Model**: A Hezar Model instance
- **Train & Eval Datasets**: Train and evaluation datasets (Hezar Dataset instances)
- **Preprocessor(s)**: Model's preprocessor if it's not already in the model (`model.preprocessor == None`)

As an example, here is how you can initialize a trainer for text classification using BERT:

```python
from hezar.models import BertTextClassification, BertTextClassificationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig


dataset_path = "hezarai/sentiment-dksf"
base_model_path = "hezarai/bert-base-fa"

train_dataset = Dataset.load(dataset_path, split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", tokenizer_path=base_model_path)

model = BertTextClassification(BertTextClassificationConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    output_dir="bert-fa-sentiment-analysis-dksf",
    task="text_classification",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
    metrics=["f1", "precision", "accuracy", "recall"],
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
```

### `Trainer.__init__()`
So what does exactly happen in the Trainer's `__init__`?

* Config sanitization
* Configure determinism (controlled by `config.seed`)
* Configure device(s)
* Setup model and
* preprocessor (load pretrained weights if configured)
* Setup data loaders
* Setup optimizer and LR scheduler
* Configure precision if set
* Setup metrics handler (chosen based on the trainer's task, type `MetricsHandler`)
* Configure paths and loggers
* Setup trainer state

Now the trainer has all the objects ready (almost!).

## Training process
The training process starts right when you call `trainer.train()`. This simple method does all the heavy lifting needed
during a full training process. We'll go through each of them one by one. 

In a nutshell, the training process is simply a repeating loop of training the model on the full train data and then
evaluating it on the evaluation data followed by calculating the metrics and saving logs and results.

### 1. Training info
Right before the trainer starts the main training process, it simply outputs some info about the run. 
These info would be something like this:
```
******************** Training Info ********************

  Output Directory: `bert-fa-sentiment-analysis-dksf`
  Task: `text_classification`
  Model: `BertTextClassification`
  Init Weights: `hezarai/bert-base-fa`
  Device(s): `cuda`
  Training Dataset: `TextClassificationDataset(path=hezarai/sentiment-dksf['train'], size=28602)`
  Evaluation Dataset: `TextClassificationDataset(path=hezarai/sentiment-dksf['test'], size=2315)`
  Optimizer: `adam`
  Initial Learning Rate: `2e-05`
  Learning Rate Decay: `0.0`
  Epochs: `5`
  Batch Size: `8`
  Number of Parameters: `118299651`
  Number of Trainable Parameters: `118299651`
  Mixed Precision: `Full (fp32)`
  Metrics: `['f1', 'precision', 'accuracy', 'recall']`
  Checkpoints Path: `bert-fa-sentiment-analysis-dksf/checkpoints`
  Logs Path: `bert-fa-sentiment-analysis-dksf/logs`

*******************************************************
```

### 2. Inner training loop
The inner training loop which is invoked by `.inner_training_loop(epoch)` trains the model on the full iteration of the
training data. This iteration itself is a repeating loop of the below processes:
1. Preparation of the input batch using `.prepare_input_batch(input_batch)` which performs all the necessary validations and
checks on the input batch like casting data type, device, etc.
2. Performing one training step using `.training_step(input_batch)` which is the casual PyTorch forward pass followed by
loss computation and backward pass and finally an optimizer step. This method outputs the loss value along the model outputs.
3. Aggregation of loss values through the training loop and live verbose of loss average up until that point in the progress bar.

### 3. Evaluation loop
The evaluation loop is also the same as the training loop, but it does everything in eval mode (`torch.inference_mode`).
1. Preparation of the input batch using `.prepare_input_batch(input_batch)` which performs all the necessary validations and
checks on the input batch like casting data type, device, etc.
2. Performing one evaluation step using `.evaluation_step(input_batch)`:
    - Perform one forward pass on the inputs
    - Calculate loss on the inputs
    - Calculate generation outputs if the model is generative (`model.is_generative`)
3. Calculate metrics on the outputs which is handled by the metrics handler (`self.metrics_handler`)
4. Aggregation of metrics values through the evaluation loop and live verbose of the averages in the progress bar. (the
final value is the total average value)

### 4. Saving & logging
When a full training loop and evaluation is done, the trainer saves some properties:
- Save the trainer state in the `trainer_state.yaml` file at `config.output_dir/config.checkpoints_dir`.
- Save the current checkpoint (model weights, configs, etc.) at `config.checkpoints_dir/epoch`.
- Save all the new training and evaluation results to the CSV file and TensorBoard at `config.logs_dir`

All the above steps are repeated for `config.num_epochs` times.


## Save & Push to Hub
Like all other components in Hezar, you can also save or push the trainer to the Hub.

### Save Trainer
In order to save the trainer, you can simply call `trainer.save()` which accepts the following:
- `path`: The target directory to save the objects in the trainer
- `config_filename`
- `model_filename`
- `model_config_filename`
- `subfolder`
- `dataset_config_file`

Overall, the `save` method saves the model, preprocessor, dataset config and the trainer config.

### Push to Hub
You can also push the trainer to the hub. This method just calls the `push_to_hub` method on the model, preprocessor, and configs.


## Advanced Training & Customization
The Trainer is implemented in a really flexible and customizable way so that any change can be done by simply overriding 
your desired method. You can learn more about how you can do such things [here](advanced_training.md)