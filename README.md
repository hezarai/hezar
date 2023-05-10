
<p align="center">
<img src="https://avatars.githubusercontent.com/u/105852828?s=200&v=4" style="border-radius: 20px;"><br>
</p>


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
model_path = "hezarai/roberta-fa-sentiment-digikala-snappfood"
# load model and tokenizer
model = Model.load(model_path)
tokenizer = Tokenizer.load(model_path)
# tokenize inputs
example = ["کتابخانه هزار، بهترین کتابخانه هوش مصنوعیه"]
inputs = tokenizer(example, return_tensors="pt")
# inference
outputs = model.predict(inputs)
# print outputs
print(outputs)
```
```commandline
{'labels': ['positive'], 'probs': [0.9960528612136841]}
```
### Build a model from scratch
```python
from hezar import build_model
# build model using its registry name
model = build_model("bert_text_classification", id2label={0: "negative", 1: "positive"})
print(model)
```
### Write your own model
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
        self.nn = nn.Linear(in_features=self.config.input_shape, out_features=self.config.output_shape)

    def forward(self, inputs: list, **kwargs):
        inputs = Tensor(inputs).reshape(1, -1)
        x = self.nn(inputs)
        return x

model = Perceptron(PerceptronConfig())
inputs = [1, 2, 3, 4]
model.predict(inputs)
```
```
tensor([[1.6096, 0.4799]])
```
As you can see, defining a new network is just like a typical PyTorch module, but instead you get access to some amazing functionalities out-of-the-box like pushing to the Hub!
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
This is a really heavy project to be maintained by a couple of developers. The idea isn't novel at all but actually doing it is really difficult hence it's the only one in the whole history of the Persian open source! So any contribution is appreciated ❤️

