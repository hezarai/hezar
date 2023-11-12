from hezar.models import T5TextGeneration, T5TextGenerationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig

dataset_path = "hezarai/xlsum-fa"
base_model_path = "hezarai/t5-base-fa"

train_dataset = Dataset.load(
    dataset_path,
    split="train",
    tokenizer_path=base_model_path,
    max_length=256,
    max_target_length=256,
)
eval_dataset = Dataset.load(
    dataset_path,
    split="test",
    tokenizer_path=base_model_path,
    max_length=256,
    max_target_length=256,
)

model = T5TextGeneration(T5TextGenerationConfig())
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    task="text_generation",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=10,
    checkpoints_dir="checkpoints/",
    metrics=["rouge"],
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
