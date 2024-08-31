from hezar.models import T5TextGeneration, T5TextGenerationConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig

if __name__ == '__main__':
    dataset_path = "hezarai/xlsum-fa"
    base_model_path = "hezarai/t5-base-fa"
    PREFIX = "این متن را خلاصه کن: "

    train_dataset = Dataset.load(
        dataset_path,
        split="train",
        preprocessor=base_model_path,
        max_length=384,
        max_target_length=80,
        prefix=PREFIX
    )
    eval_dataset = Dataset.load(
        dataset_path,
        split="test",
        preprocessor=base_model_path,
        max_length=384,
        max_target_length=80,
        prefix=PREFIX,
    )

    model = T5TextGeneration(T5TextGenerationConfig())
    preprocessor = Preprocessor.load(base_model_path)

    train_config = TrainerConfig(
        output_dir="t5-base-fa-xlsum",
        task="text_generation",
        device="cuda",
        init_weights_from=base_model_path,
        batch_size=8,
        eval_batch_size=64,
        num_epochs=10,
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
