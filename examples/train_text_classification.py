from hezar import (
    TrainerConfig,
    TextClassificationTrainer,
    Dataset,
    build_model,
    OptimizerConfig,
    LRSchedulerConfig
)


name = "distilbert_text_classification"
dataset_path = "hezarai/sentiment_digikala_snappfood"
lm_path = "hezarai/distilbert-base-fa"

train_dataset = Dataset.load(dataset_path, split="train", tokenizer_path=lm_path)
eval_dataset = Dataset.load(dataset_path, split="test", tokenizer_path=lm_path)

model = build_model(name, id2label=train_dataset.id2label)
optim_config = OptimizerConfig(name="adam", lr=2e-5, scheduler=LRSchedulerConfig(name="reduce_on_plateau"))
train_config = TrainerConfig(
    device="cuda",
    optimizer=optim_config,
    init_weights_from=lm_path,
    batch_size=8,
    num_epochs=5,
    checkpoints_dir="checkpoints/",
    metrics={"f1": {"task": "multiclass"}},
)

trainer = TextClassificationTrainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
trainer.train()
