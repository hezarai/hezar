from hezar import (
    build_optimizer,
    build_scheduler,
    build_model,
    build_dataset,
    Trainer,
    TrainConfig,
)

dataset_config = dict(
    path="hezar-ai/sentiment_digikala_snappfood",
    text_field="text",
    label_field="label",
    tokenizer_path="hezar-ai/distilbert-fa",
)
train_dataset = build_dataset(name="text_classification", split="train", **dataset_config)
eval_dataset = build_dataset(name="text_classification", split="test", **dataset_config)

model = build_model("distilbert_text_classification", num_labels=train_dataset.num_labels)
model.config.id2label = train_dataset.id2label
optimizer = build_optimizer("adamw", model.parameters(), lr=2e-5)
lr_scheduler = build_scheduler("reduce_on_plateau", optimizer=optimizer)

c = TrainConfig(
    name="distilbert_text_classification",
    device="cuda",
    batch_size=16,
    num_train_epochs=10,
    checkpoints_dir="checkpoints/",
    metrics=["f1"],
    metrics_kwargs={"task": "multiclass", "average": "micro"}
)

trainer = Trainer(
    config=c,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)
trainer.train()
