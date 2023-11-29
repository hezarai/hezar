# Quick Tour
## Models
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
model = Model.load("hezarai/crnn-fa-printed-96-long")
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
## Word Embeddings
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
For a full guide on the embeddings module, see the [embeddings tutorial](https://hezarai.github.io/hezar/tutorial/embeddings.html).
## Datasets
You can load any of the datasets on the [Hub](https://huggingface.co/hezarai) like below:
```python
from hezar.data import Dataset

sentiment_dataset = Dataset.load("hezarai/sentiment-dksf")  # A TextClassificationDataset instance
lscp_dataset = Dataset.load("hezarai/lscp-pos-500k")  # A SequenceLabelingDataset instance
xlsum_dataset = Dataset.load("hezarai/xlsum-fa")  # A TextSummarizationDataset instance
alpr_ocr_dataset = Dataset.load("hezarai/persian-license-plate-v1")  # An OCRDataset instance
...
```
The returned dataset objects from `load()` are PyTorch Dataset wrappers for specific tasks and can be used by a data loader out-of-the-box!

You can also load Hezar's datasets using 🤗Datasets:
```python
from datasets import load_dataset

dataset = load_dataset("hezarai/sentiment-dksf")
```
For a full guide on Hezar's datasets, see the [datasets tutorial](https://hezarai.github.io/hezar/tutorial/datasets.html).
## Training
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
    output_dir="bert-fa-pos-lscp-500k",
    task="sequence_labeling",
    device="cuda",
    init_weights_from=base_model_path,
    batch_size=8,
    num_epochs=5,
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

Want to go deeper? Check out the [guides](../guide/index.md).
