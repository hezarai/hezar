# Advanced Guide on Models

Models (under `hezar.models`) is the most used module in Hezar. In this section, we'll take a deeper tour of this
module.

Note that this section assumes you already know the basics of Hezar and in specific, the models module, but if not,
you can check out the introduction guide on models [here](../tutorial/models.md).

## Building Models
As you'd probably know at this point, any subclass of Model is a regular PyTorch nn.Module, so creating any model
is straightforward. But what makes it different?

First difference is in the `__init__` method. Every model has to take in a `config` parameter that contains all the
necessary parameters needed for the model to be created and initialized. This `config` parameter is a
dataclass of type `ModelConfig` derived from the base config class which is `Config`. The `Config` class is the
base config container for all configs in Hezar. Find out more about
configs [here](hezar_architecture.md/#concept-1-configurable-modules).

Take a look at the snippets below:

- **Regular PyTorch**
```python
import torch
import torch.nn as nn


class SampleCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

- **Hezar Model**
```python
from dataclasses import dataclass

import torch
import torch.nn as nn
from hezar.models import Model, ModelConfig, register_model

@dataclass
class SampleNetConfig(ModelConfig):
    name = "sample_net"
    num_channels: int = 3
    num_classes: int = 10

@register_model("sample_net", config_class=SampleNetConfig, description="My simple CNN network")
class SampleNet(Model):
    def __init__(self, config: SampleNetConfig):
        super().__init__(config=config)
        self.conv1 = nn.Conv2d(self.config.num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.config.num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
So what you actually need to do to make your PyTorch model compatible with Hezar is:
1. Move all the required arguments of the model to a new dataclass config by deriving the `ModelConfig` class
2. Implement your model by inheriting from `Model` instead of `nn.Module` and construct your model architecture by using config parameters
3. Optionally you can register your model by using the `register_model` decorator under the same `name` parameter  in the config. This step makes your model importable/exportable (compatible with `save`, `load`, `push_to_hub` methods)


## Models Registry System
Registry system is not specific to models but all modules in Hezar. For more info on registries check out [this section](hezar_architecture.md/#concept-2-modules-registries).

Registries are required for finding the right class when trying to load a model from a path (local or Hub). Each model must
have a name which must be the same as the one in its config under `name` parameter. (take a look at the example above).

To see all the available models use:
```python
from hezar.utils import list_available_models

print(list_available_models())
```

### Models Registry and `build_model`
The `models_registry` (like all registry containers in Hezar) is a dictionary of model names mapped to their module classes
and config classes. So one can easily build a model with default parameters by its registry key.

```python
from hezar.registry import models_registry

bert = models_registry["bert"].module_class(models_registry["bert"].config_class())
```
Obviously, this is so ugly and long so lets use the build method `build_model`. This method takes in 3 paramters:
- `name`: The model name which must be present in `model_registry` keys
- `config`: Optional model config
- `**kwargs`: Extra config parameters as keyword arguments that overwrites the default config parameters.
```python
from hezar.builders import build_model

bert = build_model("bert")
```
You can also pass config parameters to the `build_model` method as kwargs to overwrite default config parameters:
```python
from hezar.builders import build_model

bert = build_model("bert", hidden_size=768)
```
Or pass in the whole config to the build function:
```python
from hezar.builders import build_model
from hezar.models import BERTConfig

bert = build_model("bert", BERTConfig(hidden_act="gelu", hidden_size=840))
```

## Inference & Prediction
The end-to-end prediction for any model is done by calling the `predict()` method on raw inputs.
The `predict()` method itself, calls three main methods in order:
- `preprocess()`
- `forward()`/`generate()`*
- `post_process()`

*based on model type; regular or generative
### Preprocessing/Post-processing
These steps are performed by two methods:
- `preprocess()`: takes in raw inputs and processes them to create direct model inputs and returns a dictionary of named
inputs that is unpacked for model's `forward`/`generate` method. Each model can handle raw inputs however necessary.
But ready-to-use models in Hezar, all use preprocessor modules. preprocessor modules can be tokenizers, feature extractors,
normalizers, etc. The `Model` class has a `preprocessor` property that stores a dictionary of the required preprocessors
for the model. These preprocessors are named after their original name in config or registry like `bpe_tokenizer`, `image_processor`, etc.
- `post_process()`: responsible for converting model forward/generate outputs which are usually tensors to a human-readable
format. You might also use the `preprocessor` property at this stage i.e, for decoding, etc.

#### The `preprocessor` property
The preprocessor property can be directly set on a model. This preprocessor must be of type `Preprocessor`. If a model
needs multiple preprocessors you can pass in a dictionary of preprocessors by their name (preferably registry name).
You can use the preprocessor property like below:
```python
class TextClassificationModel(Model):
    def __init__(self):
        ...

    def forward(self, inputs):
        ...

    def preprocess(self, raw_texts):
        tokenizer = self.preprocessor["bpe_tokenizer"]
        model_inputs = tokenizer(raw_texts, return_tensors="pt")
        return model_inputs

    def post_process(self, model_outputs):
        logits = model_outputs["logits"]
        label_ids = logits.argmax(1)
        labels_str = [self.config.id2label[label_id] for label_id in label_ids]
        return labels_str
```
You can inspect the preprocessor for any model like below:
```python
from hezar.models import Model

whisper = Model.load("hezarai/whisper-small-fa")
whisper_preprocessors = whisper.preprocessor
print(whisper_preprocessors)
```
```
PreprocessorsContainer(
    [
        ('whisper_feature_extractor',
         < hezar.preprocessors.feature_extractors.audio.whisper_feature_extractor.WhisperFeatureExtractor at 0x7f6316fdcbb0 >),
        ('whisper_bpe_tokenizer',
         < hezar.preprocessors.tokenizers.whisper_bpe.WhisperBPETokenizer at 0x7f643cb13f40 >)
    ]
)
```
### Passing kwargs to `predict()`
You can also pass in additional parameters corresponding to any of the methods and the `predict()` method will figure out
how each arg should be passed to the write method (`preprocess`, `forward` or `post_process`).

Suppose you model's methods take parameters like below:
- `preprocess(raw_inputs, return_attention_mask=False)`
- `post_process(model_inputs, output_all_scores=False)`
You can pass in parameters for such model like below:
```python
model.predict(raw_inputs, return_attention_mask=True, output_all_scores=True)
```
The predict method knows which parameter corresponds to which method. (see [issue #96](https://github.com/hezarai/hezar/issues/96))


## Saving, Loading & Pushing to Hub
All Hezar models can be easily saved, loaded and pushed to hub in the same way.

### Loading Models
Loading models is done by using the `.load()` method. This method takes in the path to the desired model which can be
a path on the Hub or a path on your local disk.
```python
from hezar.models import Model

whisper = Model.load("hezarai/whisper-small-fa")
whisper.save("my-whisper")
whisper_2 = Model.load("my-whisper")
whisper_2.push_to_hub("arxyzan/whisper-small-fa")
```
Note that the preprocessors of the model will also be loaded if available when using `Model.load()`. However, you can
disable this behavior by `Model.load(path, load_preprocessor=False)`.
#### `load()` Parameters
`Model.load()` takes these parameters:
- `hub_or_local_path`: Path to a Hub repo or a folder on your local disk
- `load_locally`: Force this method to look for the path locally
- `load_preprocessor`: Whether to load the preprocessor(s) or not (defaults to True)
- `model_filename`: Optionally specify the model's weights file name (defaults to `model.pt`)
- `config_filename`: Optionally specify the model's config file name (defaults to `model_config.yaml`)
- `save_path`: Optionally save the loaded model to a custom path
- `**kwargs`: Additional config parameters to overwrite the loaded config parameters

#### Loading State Dicts
Although Hezar models are regular PyTorch `nn.Module`s, but for convenience, we overrode the `load_state_dict` in a way
that the user can load backbone models on a model for fine-tuning purposes. Also, our method can safely ignore mismatching
keys if the values are compatible. So if you receive a warning when fine-tuning a model like below:
```
Hezar (WARNING): Partially loading the weights as the model architecture and the given state dict are incompatible!
Ignore this warning in case you plan on fine-tuning this model
Incompatible keys: []
Missing keys: ['classifier.weight', 'classifier.bias']
```
You are good to go with your training because only the last classifier weights are missing and new for the training.

### Saving Models
Saving models to a path is pretty simple. Note that this method takes a **folder** path not a file path because it saves
all the files for the model, config and preprocessors to this path but instead you can control the behavior of this method
too.
#### `save()` Parameters
`Model.save()` takes these parameters:
- `path`: A path to a local folder
- `filename`: Model's file name (defaults to `model.pt`)
- `save_preprocessor`: Whether to save the preprocessor or not
- `config_filename`: Model's config file name (defaults to `model_config.yaml`)

### Pushing to the Hub
Pushing Hezar models to the Hub (just like other modules in Hezar) is done by using the `push_to_hub` method.
#### `push_to_hub()` Parameters
This method is actually the save method that is followed by the upload operation so its parameters are similar to `save`.
- `repo_id`: Path to the repo id on the Hugging Face Hub
- `filename`: Model's file name (defaults to `model.pt`)
- `config_filename`: Optionally specify the model's config file name (defaults to `model_config.yaml`)
- `push_preprocessor`: Whether to push the preprocessor or not
- `commit_message`: Commit message for this push
- `private`: Specify if the repo should be private or not. Only applicable if the repo does not already exist.

## Wrap Up
In this guide, we walked through the detail and internals of the models in Hezar. Hezar models are PyTorch Modules equiped
with extra functionalities for better integration and exportability.
