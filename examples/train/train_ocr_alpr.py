from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import Preprocessor
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig

base_model_path = "hezarai/crnn-fa-printed-96-long"
preprocessor = Preprocessor.load(base_model_path)

train_dataset = Dataset.load(
    "hezarai/persian-license-plate-v1",
    split="train",
    max_length=8,
    reverse_digits=True,
    image_processor_config=preprocessor.config,
)
eval_dataset = Dataset.load(
    "hezarai/persian-license-plate-v1",
    split="test",
    max_length=8,
    reverse_digits=True,
    image_processor_config=preprocessor.config,
)

model_config = CRNNImage2TextConfig.load(base_model_path, id2label=train_dataset.config.id2label)
model = CRNNImage2Text(model_config)

train_config = TrainerConfig(
    output_dir="crnn-fa-plate-v2",
    task="image2text",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=20,
    log_steps=100,
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
