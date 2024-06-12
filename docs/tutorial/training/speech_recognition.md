# Speech Recognition
In this tutorial, we'll finetune the Whisper model on the Persian portion of Common Voice.

Note that this model is still big and requires at least 12 GB of VRAM to train.

Let's import the required stuff:
```python
from hezar.models import Model
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig
```
Define the base model path:
```python
base_model_path = "hezarai/whisper-small"
```
## Dataset
As mentioned, we'll use the CommonVoice (Persian samples) dataset which is provided in Hezar's Hugging Face.

```python
dataset_path = "hezarai/common-voice-13-fa"

train_dataset = Dataset.load(dataset_path, preprocessor=base_model_path, split="train", labels_max_length=64)
eval_dataset = Dataset.load(dataset_path, preprocessor=base_model_path, split="test", labels_max_length=64)
```

## Model
We'll load the model (with its preprocessors) from the base model's path.
```python
model = Model.load(base_model_path)
```

## Training
Let's configure the trainer using the `TrainerConfig`:
```python
train_config = TrainerConfig(
    output_dir="whisper-small-fa-commonvoice",
    task="speech_recognition",
    mixed_precision="bf16",
    resume_from_checkpoint=True,
    gradient_accumulation_steps=8,
    batch_size=4,
    log_steps=100,
    save_steps=1000,
    num_epochs=5,
    metrics=["cer", "wer"],
)
```
- Since the model is big and larger batch sizes might lead to GPU OOM, we define a `gradient_accumulation_steps` of 8.
- To reduce memory usage, we set the mixed precision to BFloat16 (`bf16`)
- Saving in between steps is recommended for easier training resumption.
- The training loss moving average is logged every 100 steps. (saved to Tensorboard)

Let's create the Trainer and start it!
```python
trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome ASR model!")
```