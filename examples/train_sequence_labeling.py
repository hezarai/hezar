from hezar import (
    TrainerConfig,
    Trainer,
    Dataset,
    build_model,
)

name = "bert_sequence_labeling"

train_dataset = Dataset.load("hezarai/lscp-500k", split="train", tokenizer_path="hezarai/bert-base-fa")
eval_dataset = Dataset.load("hezarai/lscp-500k", split="test", tokenizer_path="hezarai/bert-base-fa")

model = build_model(name, id2label=train_dataset.id2label)

train_config = TrainerConfig(
    name=name,
    device="cuda",
    optimizer={"name": "adam", "lr": 2e-5, "scheduler": {"name": "reduce_on_plateau"}},
    init_weights_from="hezarai/bert-base-fa",
    batch_size=8,
    num_epochs=5,
    checkpoints_dir="checkpoints/",
    metrics={"f1": {"task": "multiclass"}},
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
trainer.train()