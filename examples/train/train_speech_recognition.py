from hezar.models import WhisperSpeechRecognition, WhisperSpeechRecognitionConfig
from hezar.data import Dataset
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig


dataset_path = "hezarai/commonvoice-13-fa"
base_model_path = "hezarai/whisper-small-fa"

train_dataset = Dataset.load(dataset_path, split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", tokenizer_path=base_model_path)

model = WhisperSpeechRecognition(WhisperSpeechRecognitionConfig())
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    output_dir="whisper-small-fa-commonvoice",
    task="speech_recognition",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
    metrics=["cer"],
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
