# -*- coding: utf-8 -*-
from hezar import (BertTextClassification, BertTextClassificationConfig,
                   Dataset, Preprocessor, TextClassificationTrainer,
                   TrainerConfig)

dataset_path = "hezarai/sentiment-dksf"
base_model_path = "hezarai/bert-base-fa"

train_dataset = Dataset.load(dataset_path, split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", tokenizer_path=base_model_path)

model = BertTextClassification(BertTextClassificationConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
    checkpoints_dir="checkpoints/",
    metrics=["f1", "precision", "accuracy", "recall"],
)

trainer = TextClassificationTrainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
trainer.train()
