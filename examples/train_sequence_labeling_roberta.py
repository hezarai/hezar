from hezar import (
    TrainerConfig,
    SequenceLabelingTrainer,
    OptimizerConfig,
    LRSchedulerConfig,
    Dataset,
    Preprocessor,
    build_model,
)

name = "roberta_sequence_labeling"
lm_path = "hezarai/roberta-base-fa"

train_dataset = Dataset.load("hezarai/lscp-pos-500k", split="train", tokenizer_path=lm_path)
eval_dataset = Dataset.load("hezarai/lscp-pos-500k", split="test", tokenizer_path=lm_path)

model = build_model(name, id2label=train_dataset.config.id2label)
preprocessor = Preprocessor.load(lm_path)
optimizer_config = OptimizerConfig(name="adam", lr=2e-5, scheduler=LRSchedulerConfig(name="reduce_on_plateau"))
train_config = TrainerConfig(
    device="cuda",
    optimizer=optimizer_config,
    init_weights_from=lm_path,
    num_dataloader_workers=4,
    batch_size=32,
    num_epochs=3,
    checkpoints_dir="checkpoints/",
    metrics=["seqeval"],
)

trainer = SequenceLabelingTrainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
trainer.train()
trainer.push_to_hub("hezarai/roberta-fa-pos-lscp-500k")
