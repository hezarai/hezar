from hezar.models import Model
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig


base_model_path = "hezarai/vit-roberta-fa-base"

train_dataset = Dataset.load("hezarai/flickr30k-fa", split="train", tokenizer_path=base_model_path, max_length=128)
eval_dataset = Dataset.load("hezarai/flickr30k-fa", split="test", tokenizer_path=base_model_path, max_length=128)

model = Model.load(base_model_path)

train_config = TrainerConfig(
    output_dir="vit-roberta-fa-image-captioning-flickr30k",
    task="image2text",
    device="cuda",
    batch_size=16,
    num_epochs=20,
    use_amp=True,
    metrics=["wer"],
    metric_for_best_model="wer"
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
trainer.train()
