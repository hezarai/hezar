# Dataset Processors
Initially, Hezar's Trainer worked only with PyTorch Datasets (derived from `torch.utils.data.Dataset`) like all Hezar datasets classes
at `hezar.data.datasets`. Moving on, we also added support for any iterable as the dataset in Hezar's Trainer.

One really important type of datasets is 🤗 Datasets. The Trainer almost supported these type of datasets since day one,
but implementing the data pipelines must have been handled by the user. That's why Hezar (`v0.42.0>=`) added a new category of classes
called Dataset Processors. These classes are used as dataset map callables which has the following benefits:
- The same processing pipeline in the corresponding `hezar.data.Dataset` subclass is implemented as a map function.
For example, `SpeechRecognitionDatasetProcessor` corresponds to `SpeechRecognitionDataset`. 
- Features like cacheing, multiprocessing, batch processing, etc. are now available since objects are of type `datasets.Dataset`.
- Other dataset processing pipelines from other codes feel like plug-and-play to work with Hezar's `Trainer`.

Now lets see an example demonstrating both cases:

**Classic 🤗Datasets**

Here we need to implement a map function that processes our samples. 🤗Datasets `map` function works on callables that
operate on either single or batched inputs. Below is an implementation for batched processing:
```python
from datasets import load_dataset, Audio
from hezar.preprocessors import Preprocessor


preprocesssor = Preprocessor.load("hezarai/whisper-small-fa")
feature_extractor = preprocesssor.audio_feature_extractor
tokenizer = preprocesssor.tokenizer

def batch_process_fn(data):
    # Extract audio arrays and transcripts
    audio_arrays = data["audio"]  # Assuming audio arrays are stored under the "audio" key
    transcripts = data["transcript"]  # Assuming transcripts are stored under the "transcript" key

    # Extract input features in batch
    input_features = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="np",  # Return as numpy for compatibility with map
    )["input_features"]

    # Tokenize transcripts in batch
    labels = tokenizer(
        transcripts,
        padding="max_length",
        max_length=448,
        return_tensors="np",
    )

    # Add processed data to the dictionary
    data["input_features"] = input_features
    data["labels"] = labels["input_ids"]
    data["attention_mask"] = labels["attention_mask"]

    return data

dataset = load_dataset("hezarai/common-voice-13-fa", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.select_columns(["sentence", "audio"])
# Apply the function to the dataset using map
processed_dataset = dataset.map(batch_process_fn, batched=True)
processed_dataset = processed_dataset.select_columns(["input_features", "labels", "attention_mask"])
print(processed_dataset[0])
```

**Hezar Dataset Processors**

Here's an equivalent code using the `SpeechRecognitionDatasetProcessor` that has implemented the same map function as a
callable (`SpeechRecognitionDatasetProcessor.__call__()`) that works with both single and batched inputs out of the box!
```python
from datasets import load_dataset, Audio

from hezar.data import SpeechRecognitionDatasetProcessor, SpeechRecognitionDataCollator
from hezar.preprocessors import Preprocessor

dataset = load_dataset("hezarai/common-voice-13-fa", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.select_columns(["sentence", "audio"])

preprocesssor = Preprocessor.load("hezarai/whisper-small-fa")

dataset_processor = SpeechRecognitionDatasetProcessor(
    tokenizer=preprocesssor.tokenizer,
    feature_extractor=preprocesssor.audio_feature_extractor,
    transcript_column="sentence",
    audio_array_padding="max_length",
)
data_collator = SpeechRecognitionDataCollator(
    feature_extractor=preprocesssor.audio_feature_extractor,
    tokenizer=preprocesssor.tokenizer,
    labels_padding="max_length",
    labels_max_length=256,
)
processed_dataset = dataset.map(
    dataset_processor,
    batched=True,
    batch_size=100,
    desc="Processing dataset..."
)
processed_dataset = processed_dataset.select_columns(["input_features", "labels", "attention_mask"])
print(processed_dataset[0])
```

## How Dataset Processors Work
Dataset processors classes are callable classes that receive dataset rows/batches and process them when used as a map function
with `datasets.Dataset.map()`. Here are the current supported dataset processors:
- `ImageCaptioningDatasetProcessor`
- `OCRDatasetProcessor`
- `SequenceLabelingDatasetProcessor`
- `SpeechRecognitionDatasetProcessor`
- `TextClassificationDatasetProcessor`
- `TextSummarizationDatasetProcessor`

All the above classes inherit from the base `DatasetProcessor` class and must implement the following two methods:
- `process_single(data, **kwargs)`
- `process_batch(data, **kwargs)`

The main `__call__()` method is implemented in the base class to figure out if the input `data` is a single row or a batch.


## A Training Example
Let's see how we can use a dataset processor to load and process a Hub dataset for speech recognition and train a Whisper model.

```python
from datasets import load_dataset, Audio

from hezar.data import SpeechRecognitionDatasetProcessor, SpeechRecognitionDataCollator
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig
from hezar.models import Model

dataset = load_dataset("hezarai/common-voice-13-fa", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.select_columns(["sentence", "audio"])

base_model_path = "hezarai/whisper-small"
preprocesssor = Preprocessor.load(base_model_path)

dataset_processor = SpeechRecognitionDatasetProcessor(
    tokenizer=preprocesssor.tokenizer,
    feature_extractor=preprocesssor.audio_feature_extractor,
    transcript_column="sentence",
    audio_array_padding="max_length",
)
# This is the same data collator used in `SpeechRecognitionDataset`
data_collator = SpeechRecognitionDataCollator(
    feature_extractor=preprocesssor.audio_feature_extractor,
    tokenizer=preprocesssor.tokenizer,
    labels_padding="max_length",
    labels_max_length=256,
)
processed_dataset = dataset.map(
    dataset_processor,
    batched=True,
    batch_size=100,
    desc="Processing dataset..."
)
# Select needed columns for training
processed_dataset = processed_dataset.select_columns(["input_features", "labels", "attention_mask"])
# Split dataset for train/evaluation
processed_dataset = processed_dataset.train_test_split(test_size=0.1)

model = Model.load(base_model_path)

train_config = TrainerConfig(
    output_dir="whisper-small-fa-commonvoice",
    task="speech_recognition",
    init_weights_from=base_model_path,
    mixed_precision="bf16",
    gradient_accumulation_steps=8,
    batch_size=4,
    num_epochs=5,
    metrics=["cer", "wer"],
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    data_collator=data_collator,
)
trainer.train()
```

## Wrap-up
Dataset processors are simple, yet powerful callable classes to be used for dataset processing using the `.map()` function
in 🤗Datasets. This integration means that all 🤗Dataset features are unlocked when working with Hezar!
