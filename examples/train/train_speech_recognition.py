from hezar.models import Model
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig


dataset_path = "hezarai/common-voice-13-fa"
base_model_path = "hezarai/whisper-small"

train_dataset = Dataset.load(dataset_path, split="train", preprocessor=base_model_path, labels_max_length=64)
eval_dataset = Dataset.load(dataset_path, split="test", preprocessor=base_model_path, labels_max_length=64)
model = Model.load(base_model_path)

train_config = TrainerConfig(
    output_dir="whisper-small-fa-commonvoice",
    task="speech_recognition",
    init_weights_from=base_model_path,
    mixed_precision="bf16",
    gradient_accumulation_steps=8,
    batch_size=4,
    num_epochs=5,
    metrics=["cer", "wer"],
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
trainer.train()
