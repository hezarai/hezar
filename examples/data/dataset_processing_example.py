# from datasets import load_dataset
#
# from hezar.data import ImageCaptioningDatasetProcessor
# from hezar.preprocessors import Tokenizer, ImageProcessor
#
# dataset = load_dataset("hezarai/flickr30k-fa", split="train").select(indices=list(range(4000)))
# tokenizer = Tokenizer.load("hezarai/vit-roberta-fa-image-captioning-flickr30k")
# image_processor = ImageProcessor.load("hezarai/vit-roberta-fa-image-captioning-flickr30k")
# dataset_processor = ImageCaptioningDatasetProcessor(tokenizer=tokenizer, image_processor=image_processor)
#
# processed_dataset = dataset.map(
#     dataset_processor,
#     # batched=True,
#     # batch_size=1000,
#     load_from_cache_file=False,
#     # num_proc=4,
#     desc="Processing dataset..."
# )
# processed_dataset.set_format("torch")
# print(processed_dataset[0])

#
# from datasets import load_dataset
#
# from hezar.data import TextClassificationDatasetProcessor
# from hezar.preprocessors import Tokenizer
#
# dataset = load_dataset("hezarai/sentiment-dksf", split="train")
# tokenizer = Tokenizer.load("hezarai/roberta-base-fa")
# dataset_processor = TextClassificationDatasetProcessor(tokenizer=tokenizer, padding="longest")
#
# processed_dataset = dataset.map(
#     dataset_processor,
#     batched=True,
#     batch_size=1000,
#     load_from_cache_file=False,
#     num_proc=4,
#     desc="Processing dataset..."
# )
# processed_dataset.set_format("torch")
# print(processed_dataset[0])


# from datasets import load_dataset
#
# from hezar.data import SequenceLabelingDatasetProcessor
# from hezar.preprocessors import Tokenizer
#
# dataset = load_dataset("hezarai/lscp-pos-500k", split="train")
# tokenizer = Tokenizer.load("hezarai/roberta-base-fa")
# dataset_processor = SequenceLabelingDatasetProcessor(tokenizer=tokenizer, padding="longest")
#
# processed_dataset = dataset.map(
#     dataset_processor,
#     batched=True,
#     batch_size=1000,
#     load_from_cache_file=False,
#     # num_proc=4,
#     desc="Processing dataset..."
# )
# processed_dataset.set_format("torch")
# print(processed_dataset[0])


# from datasets import load_dataset
#
# from hezar.data import TextSummarizationDatasetProcessor
# from hezar.preprocessors import Tokenizer
#
# dataset = load_dataset("hezarai/xlsum-fa", split="train")
# tokenizer = Tokenizer.load("hezarai/t5-base-fa")
# dataset_processor = TextSummarizationDatasetProcessor(tokenizer=tokenizer, padding="longest")
#
# processed_dataset = dataset.map(
#     dataset_processor,
#     # batched=True,
#     # batch_size=1000,
#     load_from_cache_file=False,
#     num_proc=10,
#     desc="Processing dataset..."
# )
# processed_dataset.set_format("torch")
# print(processed_dataset[0])

# from datasets import load_dataset
#
# from hezar.data import OCRDatasetProcessor
# from hezar.preprocessors import ImageProcessor
# from hezar.configs import ModelConfig
# from hezar.utils import is_text_valid
#
#
# dataset = load_dataset("hezarai/parsynth-ocr-200k", split="train[:3000]")
# id2label = ModelConfig.load("hezarai/crnn-fa-printed-96-long", filename="model_config.yaml")["id2label"]  # hack
#
# # Cleanup dataset
# max_length = 48
# valid_indices = []
# invalid_indices = []
# for i, sample in enumerate(list(iter(dataset))):
#     path, text = sample.values()
#     if len(text) <= max_length and is_text_valid(text, id2label.values()):
#         valid_indices.append(i)
# dataset = dataset.select(valid_indices)
#
# image_processor = ImageProcessor.load("hezarai/crnn-fa-printed-96-long")
# dataset_processor = OCRDatasetProcessor(image_processor=image_processor, id2label=id2label)
# processed_dataset = dataset.map(
#     dataset_processor,
#     # batched=True,
#     # batch_size=1000,
#     load_from_cache_file=False,
#     # num_proc=10,
#     desc="Processing dataset..."
# )
# processed_dataset.set_format("torch")
# print(processed_dataset[0])

from datasets import load_dataset
from torch.utils.data import DataLoader

from hezar.data import SpeechRecognitionDatasetProcessor, SpeechRecognitionDataCollator
from hezar.preprocessors import Tokenizer, AudioFeatureExtractor

dataset = load_dataset("parquet", split="train", data_files=["train-00001-of-00002.parquet"]).select(list(range(100)))
dataset = dataset.select_columns(["sentence", "audio"])
tokenizer = Tokenizer.load("hezarai/whisper-small-fa")
feature_extractor = AudioFeatureExtractor.load("hezarai/whisper-small-fa")
dataset_processor = SpeechRecognitionDatasetProcessor(
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    transcript_field="sentence",
    labels_padding=None,
    audio_array_padding="max_length",
)
data_collator = SpeechRecognitionDataCollator(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    labels_padding="max_length",
    labels_max_length=256,
)
processed_dataset = dataset.map(
    dataset_processor,
    batched=True,
    batch_size=100,
    load_from_cache_file=False,
    # num_proc=10,
    desc="Processing dataset..."
)
processed_dataset = processed_dataset.select_columns(["input_features", "labels", "attention_mask"])
processed_dataset.set_format("torch")
data_loader = DataLoader(processed_dataset, batch_size=16, collate_fn=data_collator)
x = next(iter(data_loader))
print(x)
