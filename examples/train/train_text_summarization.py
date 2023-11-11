from hezar.models import GPT2TextGeneration, GPT2TextGenerationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig

dataset_path = "hezarai/xlsum-fa"
base_model_path = "hezarai/gpt2-base-fa"

train_dataset = Dataset.load(dataset_path, split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", tokenizer_path=base_model_path)

model = GPT2TextGeneration(GPT2TextGenerationConfig())
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    task="text_generation",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=10,
    checkpoints_dir="checkpoints/",
    metrics=["bleu", ],
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
