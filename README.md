
<p align="center">
  <img src="https://raw.githubusercontent.com/hezarai/hezar/main/hezar.png"/>
</p>

_<p align="center"> The all-in-one AI library for Persian_
<br><br>
![PyPI Version](https://img.shields.io/pypi/v/hezar?color=blue)
[![PyPi Downloads](https://static.pepy.tech/badge/hezar)](https://pepy.tech/project/hezar)
![PyPI License](https://img.shields.io/pypi/l/hezar)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/hezarai/hezar/.github%2Fworkflows%2Fdocs-deploy.yml?label=docs)<br>
[![Static Badge](https://img.shields.io/badge/Hugging_Face_Hub-yellow?label=%F0%9F%A4%97&labelColor=yellow&link=https%3A%2F%2Fhuggingface.co%2Fhezarai)](https://huggingface.co/hezarai)
[![Static Badge](https://img.shields.io/badge/Telegram_Channel-blue?logo=telegram&link=https%3A%2F%2Ft.me%2Fhezarai)](https://t.me/hezarai)
</p>

**Hezar** (meaning **_thousand_** in Persian) is a multipurpose AI library built to make AI easy for the Persian community!

Hezar is a library that:
- brings together all the best works in AI for Persian
- makes using AI models as easy as a couple of lines of code
- seamlessly integrates with Hugging Face Hub for all of its models
- has a highly developer-friendly interface
- has a task-based model interface which is more convenient for general users.
- is packed with additional tools like word embeddings, tokenizers, feature extractors, etc.
- comes with a lot of supplementary ML tools for deployment, benchmarking, optimization, etc.
- and more!

## Installation
Hezar is available on PyPI and can be installed with pip:
```
pip install hezar
```
You can also install the latest version from the source.
Clone the repo and execute the following commands:
```
git clone https://github.com/hezarai/hezar.git
pip install ./hezar
```
## Documentation
Explore Hezar to learn more on the [docs](https://hezarai.github.io/hezar/index.html) page or explore the key concepts:
- [Getting Started](https://hezarai.github.io/hezar/get_started/overview.html)
- [Quick Tour](https://hezarai.github.io/hezar/get_started/quick_tour.html)
- [Tutorials](https://hezarai.github.io/hezar/tutorial/models.html)
- [Developer Guides](https://hezarai.github.io/hezar/guide/hezar_architecture.html)
- [Contribution](https://hezarai.github.io/hezar/contribute/contribute_to_hezar.html)
- [Reference API](https://hezarai.github.io/hezar/source/index.html)

## Quick Tour
### Models
There's a bunch of ready to use trained models for different tasks on the Hub!

**🤗Hugging Face Hub Page**: [https://huggingface.co/hezarai](https://huggingface.co/hezarai)

Let's walk you through some examples!

- **Text Classification (sentiment analysis, categorization, etc)**
```python
from hezar.models import Model

example = ["هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"]
model = Model.load("hezarai/bert-fa-sentiment-dksf")
outputs = model.predict(example)
print(outputs)
```
```
[[{'label': 'positive', 'score': 0.812910258769989}]]
```
- **Sequence Labeling (POS, NER, etc.)**
```python
from hezar.models import Model

pos_model = Model.load("hezarai/bert-fa-pos-lscp-500k")  # Part-of-speech
ner_model = Model.load("hezarai/bert-fa-ner-arman")  # Named entity recognition
inputs = ["شرکت هوش مصنوعی هزار"]
pos_outputs = pos_model.predict(inputs)
ner_outputs = ner_model.predict(inputs)
print(f"POS: {pos_outputs}")
print(f"NER: {ner_outputs}")
```
```
POS: [[{'token': 'شرکت', 'label': 'Ne'}, {'token': 'هوش', 'label': 'Ne'}, {'token': 'مصنوعی', 'label': 'AJe'}, {'token': 'هزار', 'label': 'NUM'}]]
NER: [[{'token': 'شرکت', 'label': 'B-org'}, {'token': 'هوش', 'label': 'I-org'}, {'token': 'مصنوعی', 'label': 'I-org'}, {'token': 'هزار', 'label': 'I-org'}]]
```
- **Language Modeling (Mask Filling)**
```python
from hezar.models import Model

roberta_mlm = Model.load("hezarai/roberta-fa-mlm")
inputs = ["سلام بچه ها حالتون <mask>"]
outputs = roberta_mlm.predict(inputs, top_k=1)
print(outputs)
```
```
[[{'token': 'چطوره', 'sequence': 'سلام بچه ها حالتون چطوره', 'token_id': 34505, 'score': 0.2230483442544937}]]
```
- **Speech Recognition**
```python
from hezar.models import Model

whisper = Model.load("hezarai/whisper-small-fa")
transcripts = whisper.predict("examples/assets/speech_example.mp3")
print(transcripts)
```
```
[{'text': 'و این تنها محدود به محیط کار نیست'}]
```
- **Image to Text (OCR)**
```python
from hezar.models import Model
# OCR with TrOCR
model = Model.load("hezarai/trocr-base-fa-v2")
texts = model.predict(["examples/assets/ocr_example.jpg"])
print(f"TrOCR Output: {texts}")

# OCR with CRNN
model = Model.load("hezarai/crnn-base-fa-64x256")
texts = model.predict("examples/assets/ocr_example.jpg")
print(f"CRNN Output: {texts}")
```
```
TrOCR Output: [{'text': 'چه میشه کرد، باید صبر کنیم'}]
CRNN Output: [{'text': 'چه میشه کرد، باید صبر کنیم'}]
```
![](https://raw.githubusercontent.com/hezarai/hezar/main/examples/assets/ocr_example.jpg)

- **Image to Text (License Plate Recognition)**
```python
from hezar.models import Model

model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")
plate_text = model.predict("assets/license_plate_ocr_example.jpg")
print(plate_text)  # Persian text of mixed numbers and characters might not show correctly in the console
```
```
[{'text': '۵۷س۷۷۹۷۷'}]
```
![](https://raw.githubusercontent.com/hezarai/hezar/main/examples/assets/license_plate_ocr_example.jpg)

- **Image to Text (Image Captioning)**
```python
from hezar.models import Model

model = Model.load("hezarai/vit-roberta-fa-image-captioning-flickr30k")
texts = model.predict("examples/assets/image_captioning_example.jpg")
print(texts)
```
```
[{'text': 'سگی با توپ تنیس در دهانش می دود.'}]
```
![](https://raw.githubusercontent.com/hezarai/hezar/main/examples/assets/image_captioning_example.jpg)

We constantly keep working on adding and training new models and this section will hopefully be expanding over time ;)
### Word Embeddings
- **FastText**
```python
from hezar.embeddings import Embedding

fasttext = Embedding.load("hezarai/fasttext-fa-300")
most_similar = fasttext.most_similar("هزار")
print(most_similar)
```
```
[{'score': 0.7579, 'word': 'میلیون'},
 {'score': 0.6943, 'word': '21هزار'},
 {'score': 0.6861, 'word': 'میلیارد'},
 {'score': 0.6825, 'word': '26هزار'},
 {'score': 0.6803, 'word': '٣هزار'}]
```
- **Word2Vec (Skip-gram)**
```python
from hezar.embeddings import Embedding

word2vec = Embedding.load("hezarai/word2vec-skipgram-fa-wikipedia")
most_similar = word2vec.most_similar("هزار")
print(most_similar)
```
```
[{'score': 0.7885, 'word': 'چهارهزار'},
 {'score': 0.7788, 'word': '۱۰هزار'},
 {'score': 0.7727, 'word': 'دویست'},
 {'score': 0.7679, 'word': 'میلیون'},
 {'score': 0.7602, 'word': 'پانصد'}]
```
- **Word2Vec (CBOW)**
```python
from hezar.embeddings import Embedding

word2vec = Embedding.load("hezarai/word2vec-cbow-fa-wikipedia")
most_similar = word2vec.most_similar("هزار")
print(most_similar)
```
```
[{'score': 0.7407, 'word': 'دویست'},
 {'score': 0.7400, 'word': 'میلیون'},
 {'score': 0.7326, 'word': 'صد'},
 {'score': 0.7276, 'word': 'پانصد'},
 {'score': 0.7011, 'word': 'سیصد'}]
```
### Datasets
You can load any of the datasets on the [Hub](https://huggingface.co/hezarai) like below:
```python
from hezar.data import Dataset

sentiment_dataset = Dataset.load("hezarai/sentiment-dksf")  # A TextClassificationDataset instance
lscp_dataset = Dataset.load("hezarai/lscp-pos-500k")  # A SequenceLabelingDataset instance
xlsum_dataset = Dataset.load("hezarai/xlsum-fa")  # A TextSummarizationDataset instance
...
```
### Training
Hezar makes it super easy to train models using out-of-the-box models and datasets provided in the library.

```python
from hezar.models import BertSequenceLabeling, BertSequenceLabelingConfig
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig
from hezar.preprocessors import Preprocessor

base_model_path = "hezarai/bert-base-fa"
dataset_path = "hezarai/lscp-pos-500k"

train_dataset = Dataset.load(dataset_path, split="train", tokenizer_path=base_model_path)
eval_dataset = Dataset.load(dataset_path, split="test", tokenizer_path=base_model_path)

model = BertSequenceLabeling(BertSequenceLabelingConfig(id2label=train_dataset.config.id2label))
preprocessor = Preprocessor.load(base_model_path)

train_config = TrainerConfig(
    task="sequence_labeling",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
    checkpoints_dir="checkpoints/",
    metrics=["seqeval"],
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

trainer.push_to_hub("bert-fa-pos-lscp-500k")  # push model, config, preprocessor, trainer files and configs
```
You can actually go way deeper with the trainers. Refer to the [notebooks](notebooks) to see the examples!

## Going Deeper
Hezar's primary focus is on providing ready to use models (implementations & pretrained weights) for different casual tasks
without reinventing the wheel, but by being built on top of
**[PyTorch](https://github.com/pytorch/pytorch),
🤗[Transformers](https://github.com/huggingface/transformers),
🤗[Tokenizers](https://github.com/huggingface/tokenizers),
🤗[Datasets](https://github.com/huggingface/datasets),
[Scikit-learn](https://github.com/scikit-learn/scikit-learn),
[Gensim](https://github.com/RaRe-Technologies/gensim),** etc.
Besides, it's deeply integrated with the **🤗[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)** and
almost any module e.g, models, datasets, preprocessors, trainers, etc. can be uploaded to or downloaded from the Hub!

More specifically, here's a simple summary of the core modules in Hezar:
- **Models**:  Every model is a `hezar.models.Model` instance which is in fact, a PyTorch `nn.Module` wrapper with extra features for saving, loading, exporting, etc.
- **Datasets**: Every dataset is a `hezar.data.Dataset` instance which is a PyTorch Dataset implemented specifically for each task that can load the data files from the Hugging Face Hub.
- **Preprocessors**: All preprocessors are preferably backed by a robust library like Tokenizers, pillow, etc.
- **Embeddings**: All embeddings are developed on top of Gensim and can be easily loaded from the Hub and used in just 2 lines of code!
- **Trainer**: Trainer is the base class for training almost any model in Hezar or even your own custom models backed by Hezar. The Trainer comes with a lot of features and is also exportable to the Hub!
- **Metrics**: Metrics are also another configurable and portable modules backed by Scikit-learn, seqeval, etc. and can be easily used in the trainers!

For more info, check the [tutorials](https://hezarai.github.io/hezar/tutorial/)

## Contribution
This is a really heavy project to be maintained by a couple of developers.
The idea isn't novel at all but actually doing it is really difficult hence being the only one in the whole history of the Persian open source!
So any contribution is appreciated ❤️
