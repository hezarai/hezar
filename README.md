
![](hezar.png)

_<p align="center"> A seamless AI library for Persian</p>_

**Hezar** (meaning **_thousand_** in Persian) is a multipurpose AI library built to make AI easy for the Persian community!

Hezar is a library that:
- brings together all the best works in AI for Persian
- makes using AI models as easy as a couple of lines of code
- seamlessly integrates with HuggingFace Hub for all of its models
- has a highly developer-friendly interface
- has a task-based model interface which is more convenient for general users.
- is packed with additional tools like word embeddings, tokenizers, feature extractors, etc.
- comes with a lot of supplementary ML tools for deployment, benchmarking, optimization, etc.
- and more!

## Installation
Hezar is available on PyPI and can be installed with pip:
```commandline
pip install hezar
```
You can also install the latest version from the source.
Clone the repo and execute the following commands:
```commandline
git clone https://github.com/hezarai/hezar.git
pip install ./hezar
```

## Quick Tour
### Ready-to-use models from Hub
There's a bunch of ready-to-use trained models for different tasks on the Hub. See them [here](https://huggingface.co/hezarai)!

For example, you can grab a BERT-based model for sentiment analysis like so: 
```python
from hezar import Model, Tokenizer

# this is our Hub repo
model_path = "hezarai/bert-fa-sentiment-digikala-snappfood"
# load model and tokenizer
model = Model.load(model_path)
tokenizer = Tokenizer.load(model_path)
# tokenize inputs
example = ["هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"]
inputs = tokenizer(example, return_tensors="pt")
# inference
outputs = model.predict(inputs)
# print outputs
print(outputs)
```
```commandline
{'labels': ['positive'], 'probs': [0.812910258769989]}
```
### Build models from scratch
Wanna use models without any pretrained weights? Easy!

Build a raw BERT-based model for text classification with a single line of code!
```python
from hezar import build_model

model = build_model("bert_text_classification", id2label={0: "negative", 1: "positive"})
print(model)
```
You can also import model directly:
```python
from hezar import BertTextClassification, BertTextClassificationConfig

bert_tc = BertTextClassification(BertTextClassificationConfig(num_labels=2))
```
### Write your own model
It's fairly easy to extend this library or add your own model. Hezar has its own `Model` base class that is simply a normal PyTorch `nn.Module` but with some extra features!

Here's a simple example:
```python
from dataclasses import dataclass

from torch import Tensor, nn

from hezar import Model, ModelConfig


@dataclass
class PerceptronConfig(ModelConfig):
    name: str = "perceptron"
    input_shape: int = 4
    output_shape: int = 2


class Perceptron(Model):
    """
    A simple single layer network
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.nn = nn.Linear(
            in_features=self.config.input_shape,
            out_features=self.config.output_shape,
        )

    def forward(self, inputs: list, **kwargs):
        inputs = Tensor(inputs).reshape(1, -1)
        x = self.nn(inputs)
        return x

    def post_process(self, inputs, **kwargs):
        # post-process forward outputs (optional method)
        return inputs.numpy()  # convert torch tensor to numpy array


model = Perceptron(PerceptronConfig())
inputs = [1, 2, 3, 4]
outputs = model.predict(inputs)
print(outputs)
```
```
[[-0.13248837  0.7039478 ]]
```
As you can see, defining a new model is just like a typical PyTorch module, but comes with some amazing functionalities out-of-the-box like pushing to the Hub!
```python
hub_repo = "<your_hf_username>/my-awesome-perceptron"
model.push_to_hub(hub_repo)
```
```
INFO: Uploaded:`PerceptronConfig(name=preceptron)` --> `your_hf_username/my-awesome-perceptron/model_config.yaml`
INFO: Uploaded: `Perceptron(name=preceptron)` --> `your_hf_username/my-awesome-perceptron/model.pt`
```

## Documentation
Refer to the [docs](docs) for a full documentation.

## Contribution
This is a really heavy project to be maintained by a couple of developers. The idea isn't novel at all but actually doing it is really difficult hence being the only one in the whole history of the Persian open source! So any contribution is appreciated ❤️

