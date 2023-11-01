# Models
In Hezar, models are the typical PyTorch modules with some extra features for loading, saving, exporting, etc.
Let's dive into some of the most important ones!

## Models Basics
### Building Models
Like any other package, you can import any model from `hezar.models` that you want.
```python
from hezar.models import BertLM, BertLMConfig

bert = BertLM(BertLMConfig())
```
You can also configure the architecture by changing the properties in a model's config like so:
```python
config = BertLMConfig(num_hidden_layers=8, num_attention_heads=8)
bert = BertLM(config)
```

Every model in Hezar, can be pushed to or downloaded from the Hub.

### Loading pre-trained models
Loading a model from Hub is as easy as:
```python
from hezar.models import Model

bert = Model.load("hezarai/bert-base-fa")
```
The `load` methods takes the following steps to build the model:

1. Load the config file `model_config.yaml` and figure out the model's class using the `name` config parameter. (`bert_lm` in this snippet)
2. Build the model with random weights from the corresponding class. (`BertLM` in this snippet)
3. Download the weights file (`model.pt`) and load the state dict into to the model.
4. If the path contains any preprocessor, the preprocessor (`WordPieceTokenizer` in this snippet) will be loaded too.
(You can disable loading preprocessors by setting `Model.load(path, load_preprocessor=False)`)

### Inference & Prediction
Now that you have loaded a model along with its preprocessors, feature extractors, etc. you can perform an end-to-end
inference in a single line of code using `Model.predict` method.

A sequence labeling example would be like this:
```python
from hezar.models import Model

pos_model = Model.load("hezarai/bert-fa-pos-lscp-500k")  # Part-of-speech
inputs = ["شرکت هوش مصنوعی هزار"]
pos_outputs = pos_model.predict(inputs)
print(f"POS: {pos_outputs}")
```
```
POS: [[{'token': 'شرکت', 'tag': 'Ne'}, {'token': 'هوش', 'tag': 'Ne'}, {'token': 'مصنوعی', 'tag': 'AJe'}, {'token': 'هزار', 'tag': 'NUM'}]]
```

### Saving Models
You can save any model along with its config and preprocessor and other files on disk like:
```python
from hezar.models import RobertaLM, RobertaLMConfig

roberta = RobertaLM(RobertaLMConfig(vocab_size=60000))
roberta.save("my-roberta")
```

### Pushing to the Hub
Every model can be pushed to the Hub.
```python
from hezar.models import RobertaTextClassification, RobertaTextClassificationConfig

roberta = RobertaTextClassification(RobertaTextClassificationConfig(num_labels=2))
roberta.push_to_hub("arxyzan/roberta-sentiment")
```
```
INFO: Uploaded:`RobertaTextClassificationConfig(name=roberta_text_classification)` --> `arxyzan/roberta-sentiment/model_config.yaml`
INFO: Uploaded: `RobertaTextClassification(name=roberta_text_classification)` --> `arxyzan/roberta-sentiment/model.pt`
```
## Custom Models
Every Hezar model is a subclass of the base model class `Model` and the `Model` itself is a subclass of PyTorch `nn.Module`
with some extra features. So if you're familiar with PyTorch, this should feel like home!

### A Sample Perceptron
```python
from dataclasses import dataclass

from torch import Tensor, nn

from hezar.models import Model, ModelConfig
from hezar.registry import register_model


@dataclass
class PerceptronConfig(ModelConfig):
    name = "perceptron"
    input_shape: int = 4
    output_shape: int = 2


@register_model("perceptron", config_class=PerceptronConfig)
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

    def post_process(self, model_outputs, **kwargs):
        return model_outputs.numpy()

```
The only point here is that you have to pass a `ModelConfig` to your model and read everything from the config and the
rest is just typical PyTorch stuff.

Now you have access to all the features of a Hezar model.
```python
model = Perceptron(PerceptronConfig())
inputs = [1, 2, 3, 4]
outputs = model.predict(inputs)
print(outputs)
model.save("my-perceptron")
model.push_to_hub("hezarai/perceptron")
```
```
[[-1.0953112 -1.9854667]]
INFO: Uploaded:`PerceptronConfig(name=perceptron)` --> `hezarai/perceptron/model_config.yaml`
INFO: Uploaded: `Perceptron(name=perceptron)` --> `hezarai/perceptron/model.pt`
```


To learn more about the internals of the models in Hezar take a look at [the models in-depth guide](../guide/models_advanced.md)
