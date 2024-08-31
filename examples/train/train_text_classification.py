from hezar.models import BertTextClassification, BertTextClassificationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig

dataset_path = "hezarai/sentiment-dksf"
base_model_path = "hezarai/bert-base-fa"

train_dataset = Dataset.load(dataset_path, split="train", preprocessor=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", preprocessor=base_model_path)

model = BertTextClassification(BertTextClassificationConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    output_dir="bert-fa-sentiment-analysis-dksf",
    task="text_classification",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
    metrics=["f1", "precision", "accuracy", "recall"],
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
trainer.train()
