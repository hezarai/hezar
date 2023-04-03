# Hezar: A seamless AI library for Persian

Hezar is a multi-purpose AI library built to make AI easy for the Persian community!

Hezar is a library that:
- brings together all the best works in AI for Persian
- makes using AI models as easy as a couple of lines of code
- seamlessly integrates with HuggingFace Hub for all of its models
- has a highly developer-friendly interface
- comes with a lot of supplementary ML tools for deployment, benchmarking, optimization, etc.
- and more!

## Installation
Clone the repo and run:
```commandline
git clone https://github.com/hezarai/hezar.git
pip install ./hezar
```

## Quick Tour
### Use a model from Hub
```python
from hezar import Model, Tokenizer

# this is our Hub repo
path = "hezarai/bert-fa-sentiment-digikala-snappfood"
# load model and tokenizer
model = Model.load(path)
tokenizer = Tokenizer.load(path)
# tokenize inputs
inputs = tokenizer(["یه تست خیلی خفن"], return_tensors="pt")
# inference
outputs = model.predict(inputs)
# print outputs
print(outputs)
```
```commandline
{'labels': ['positive'], 'probs': [0.9629115462303162]}
```
### Build a model from scratch
```python
from hezar import build_model
# build model using its registry name
model = build_model("bert_text_classification", id2label={0: "negative", 1: "positive"})
print(model)
```

## Supported models
Hezar currently supports these models
- Text Classification
  - BERT
  - DistilBERT
- Language Modeling
  - BERT
  - DistilBERT
  - RoBERTa
- OCR
  - CRNN
  - CRAFT
  - CTPN

## Documentation
Refer to the [docs](docs) for a full documentation.

## Contribution
This is a really heavy project to be maintained by a couple of developers. The idea isn't novel at all but actually doing it is really difficult hence it's the only one in the whole history of the Persian open source! So any contribution is appreciated <3 