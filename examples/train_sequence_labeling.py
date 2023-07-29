from hezar import (
    TrainerConfig,
    SequenceLabelingTrainer,
    OptimizerConfig,
    LRSchedulerConfig,
    Dataset,
    build_model,
)

name = "bert_sequence_labeling"

train_dataset = Dataset.load("hezarai/lscp-500k", split="train", tokenizer_path="hezarai/bert-base-fa")
eval_dataset = Dataset.load("hezarai/lscp-500k", split="test", tokenizer_path="hezarai/bert-base-fa")

model = build_model(name, id2label=train_dataset.id2label)
optimizer_config = OptimizerConfig(name="adam", lr=2e-5, scheduler=LRSchedulerConfig(name="reduce_on_plateau"))
train_config = TrainerConfig(
    device="cuda",
    optimizer=optimizer_config,
    init_weights_from="hezarai/bert-base-fa",
    batch_size=8,
    num_epochs=5,
    checkpoints_dir="checkpoints/",
    metrics=["seqeval"],
)

trainer = SequenceLabelingTrainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
trainer.train()
