from hezar.data import Dataset
from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import ImageProcessor
from hezar.trainer import Trainer, TrainerConfig

train_dataset = Dataset.load("hezarai/parsynth-ocr-200k", split="train")
eval_dataset = Dataset.load("hezarai/parsynth-ocr-200k", split="test")

model = CRNNImage2Text(
    CRNNImage2TextConfig(
        id2label=train_dataset.config.id2label,
        map2seq_in_dim=1024,
        map2seq_out_dim=96
    )
)
preprocessor = ImageProcessor(train_dataset.config.image_processor_config)

train_config = TrainerConfig(
    output_dir="crnn-plate-fa-v1",
    task="image2text",
    device="cuda",
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

