from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import Preprocessor
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig


base_model_path = "hezarai/crnn-base-fa-64x256"

train_dataset = Dataset.load("hezarai/persian-license-plate-v1", split="train[:3%]", max_length=8, reverse_text=True)
eval_dataset = Dataset.load("hezarai/persian-license-plate-v1", split="test[:10%]", max_length=8, reverse_text=True)

model = CRNNImage2Text(CRNNImage2TextConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)


train_config = TrainerConfig(
    output_dir="crnn-plate-fa-v1",
    task="image2text",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=20,
    metrics=["cer"],
    metric_for_best_model="cer"
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
