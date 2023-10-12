from hezar import (BertSequenceLabeling, BertSequenceLabelingConfig, Dataset,
                   Preprocessor, SequenceLabelingTrainer, TrainerConfig)

base_model_path = "hezarai/bert-base-fa"

train_dataset = Dataset.load("hezarai/lscp-pos-500k", split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load("hezarai/lscp-pos-500k", split="test", tokenizer_path=base_model_path)

model = BertSequenceLabeling(BertSequenceLabelingConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    device="cuda",
    init_weights_from=base_model_path,
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
    preprocessor=preprocessor,
)
trainer.train()
