# Hazer's Architecture

Right from the first lines of code, Hezar was built having **simplicity**, **modularity** and **extensibility** in mind.
Hezar has a simple yet flexible design pattern that can be seen among most of its main modules. In this guide we
demonstrate the main ideas behind the design.

Going forward, by the term _module_, we mean any main class
like `Model`, `Dataset`, `Metric`, `Trainer`, `Preprocessor`,
etc.

## Concept 1: Configurable Modules

Every single module object in Hezar can be constructed from a key-value container. This container is the module's config
which contains everything needed to build an object from that module. In order to have a portable serializable config
that can be also converted to Python code, there lies Hezar's most important class called `Config`. The `Config` class
is a simple Python dataclass that is equipped with extra methods for importing, exporting, pushing to the Hub, etc.
The `Config` class is defined in `hezar/configs.py` among other config derivatives.
Right now the config derivatives are:

- `ModelConfig`
- `DatasetConfig`
- `PreprocessorConfig`
- `TrainerConfig`
- `EmbeddingConfig`
- `MetricConfig`

So every module must have its own config inherited from `Config`. When defining a new config dataclass, one must define
a unique name (as the parameter `name`), responsible for identifying the module type that uses that config class. We'll
discuss why this `name` parameter is necessary in the registry section.

To give some examples:

Let's assume you want to write a new model class called `AwesomeModel`. The first step is to provide a config dataclass:

```python
from dataclasses import dataclass
from hezar.models import ModelConfig, Model


@dataclass
class MyAwesomeModelConfig(ModelConfig):
    name = "my_awesome_model"  # this has to be a unique name among all models configs
    my_param: str = "awesome"
    other_param: str = "more_awesome"


class MyAwesomeModel(Model):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Define the layers or any other stuff here
        ...

    def forward(self, inputs, **kwargs):
        # Casual PyTorch forward method
        ...
```

Wait, what's that name for? Why would you need to define a name for everything? The short answer is
_Hezar's registry system_. So let's dive into it!

## Concept 2: Modules' Registries

There are lots of base modules in Hezar and many of which might have dozens of subclasses, but as you might have seen by
now, almost every module can load its class using the same base class in a single line. Take a look at the below
snippets:

```python
# Load a model
from hezar.models import Model

roberta_tc = Model.load("hezarai/roberta-fa-sentiment-dksf")  # roberta_tc is a RobertaTextClassification instance
bert_pos = Model.load("hezarai/bert-fa-pos-lscp-500k")  # bert_pos is a BertSequenceLabeling instance
whisper_speech = Model.load("hezarai/whisper-small-fa")  # whisper_speech is a WhisperSpeechRecognition instance
...
# Load a dataset
from hezar.data import Dataset

sentiment_dataset = Dataset.load("hezarai/sentiment-dksf")  # A TextClassificationDataset instance
lscp_dataset = Dataset.load("hezarai/lscp-pos-500k")  # A SequenceLabelingDataset instance
xlsum_dataset = Dataset.load("hezarai/xlsum-fa")  # A TextSummarizationDataset instance
...
# Load preprocessors
from hezar.preprocessors import Preprocessor

wordpiece = Preprocessor.load("hezarai/bert-base-fa")  # A WordPieceTokenizer instance
whisper_bpe = Preprocessor.load("hezarai/whisper-small-fa")  # A WhisperBPETokenizer instance
sp_unigram_bpe = Preprocessor.load("hezarai/t5-base-fa")  # A SentencePieceUnigramTokenizer instance
...
# Load embedding
from hezar.embeddings import Embedding

fasttext = Embedding.load("hezarai/fasttext-fa-300")  # A FastText instance
word2vec = Embedding.load("hezarai/word2vec-skipgram-fa-wikipedia")  # A Word2Vec instance
...
```

So, what's going on under the hood that handles module loading and initiation?

**Registry System**

Well, there are ways to tackle this challenge, but Hezar manages this by using _a global registry_ for every module
type. These registries are simple Python dictionaries that hold the properties for every module class, module config,
etc.
The general structure is like below:

```python
# Models registry for example
models_registry = {
    "bert_text_classification": Registry(
        module_class=hezar.models.text_classification.bert.bert_text_classification.BertTextClassification,
        config_class=hezar.models.text_classification.bert.bert_text_classification_config.BertTextClassificationConfig,
        description="SOME MODEL DESCRIPTION ..."
    ),
    "AND SO ON...": Registry(...)
}
```
Each registry value is a `Registry` (data)class that has 3 properties: `config_class`, `module_class` and `description`.
- `module_class`: Holds the class object for the module. Using this property you can actually create the module object.
- `config_class`: Holds the config class and can be passed to the module class so that the module can be created.
- `description`: Holds the description of the model if given.

But how are the modules inserted into the registries? The answer is _registry class decorators_

**`register_*()` Class Decorators**

In the file `hezar/registry.py`, there are a bunch of decorator functions that fulfill the task of registering any module
into the right registry automagically!
These decorators take two parameters:
- `name`: A string name that has to be the same as the one in config
- `config_class`: The config class
- `description`: Optional description for the module

The example below demonstrates registering a model:
```python
...
from hezar.models import Model, ModelConfig

@dataclass
class MyBertConfig(ModelConfig):
    name = "my_bert"
    vocab_size: int = 1000
    hidden_size: int = 768

# Below line is all you need to add `my_bert` to `models_registry`
@register_model("my_bert", config_class=MyBertConfig)
class MyBert(Model):
    def __init__(self, config: MyBertConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, inputs, **kwargs):
        ...

```
Registry decorators currently include:
- `register_model`
- `register_preprocessor`
- `register_dataset`
- `register_embedding`
- `register_metric`
- `register_trainer`

**Getting Available Modules**

To figure out what modules are available in a registry, there are also utils for that:
```python
from hezar import utils

print(utils.list_available_models())
print(utils.list_available_preprocessors())
print(utils.list_available_datasets())
print(utils.list_available_metrics())
print(utils.list_available_embeddings())
...
```
**Creating Modules from Registry Names**

So now it's pretty easy to create modules objects using their `name`! Let's say you want to create a
BPE tokenizer. You can do it this way:
```python
from hezar.registry import preprocessors_registry

module_cls = preprocessors_registry["bpe_tokenizer"].module_class
config_cls = preprocessors_registry["bpe_tokenizer"].config_class

bpe = module_cls(config_cls())
```
Although, this is not how it's actually done in Hezar because it's long and ugly! To handle this properly we use another
internal feature of Hezar called the _builders_!


**Builders**

Using builders you can build modules from their registry names in a single line of code.
These family of functions take 3 main parameters:
- `name`: A registry key name representing that module. This name has to be present in the corresponding registry!
- `config`: Optionally you can pass a config object to control how the module is built. The config has to be of a type that the module accepts.
- `**kwargs`: Optionally you can pass config parameters as keyword arguments to override the default config. (The override priority is `kwargs` > `config` > default config)
```python
from hezar import builders

bert = builders.build_model("bert_lm", hidden_size=768, vocab_size=50000)
sp_bpe = builders.build_preprocessor("sentencepiece_bpe_tokenizer")
tc_dataset = builders.build_dataset("text_classification", path="hezarai/sentiment-dksf", tokenizer_path="hezarai/bert-base-fa")
...
```
Available builders include:
- `build_model`
- `build_dataset`
- `build_preprocessor`
- `build_embedding`
- `build_metric`

So why would you need to use builders or registries when you can import everything normally? like below:
```python
from hezar.models import WhisperSpeechRecognition, WhisperSpeechRecognitionConfig

whisper = WhisperSpeechRecognition(WhisperSpeechRecognitionConfig(max_new_tokens=400))
```
The answer is that if you want to do it in a straightforward way, you can always use the classes directly. But the fact is that everything works with
configs and a config must have at least some identifiers so that a module can be initialized from it. The main usage of
the registries is to be able to create everything from the configs! So lets slide into the next section, the Hub!

## Concept 3: Hugging Face Hub Integration
In Hezar, EVERY module can be uploaded to or downloaded from the Hugging Face Hub with ease! Modules have 3 main methods
to do so:
- `load`: A method implemented in any type of base class that loads the module from the Hub or local disk automagically!
- `save`: A method to save all the necessary files and configurations to a path on the local disk.
- `push_to_hub`: A method implemented in any type of base class that pushes all the necessary files and configurations to the Hub so that the module can be loaded from the Hub again.

**Loading**

All base modules implement their own `load` method based on their characteristics. But the first step in every load
process is loading the configuration as all the info lies there, and then any other file is loaded.
For example the class `Model` first loads its config and builds the model using `build_model` and the config parameters.
Then the state dict is loaded to the model. If the path contains preprocessor files and configs, it would load them too.
On the other hand, some simple modules like metric might just load the config to create a metric instance.
One important feature of any `load` method is that like builders, it accepts config parameters as keyword arguments so
that you can override config properties.

**Saving**

Almost every module has the `save` method implemented which is responsible for saving config and other related files to the
disk. This method takes a `path` parameter which is just the base folder path and any necessary subfolder will be created
automatically based on the module type. For example, if you save a tokenizer at path `my_tokenizer/`, the `Tokenizer`'s
`save` method will create a `preprocessor` folder and saves the `tokenizer.json` and `tokenizer_config.yaml` on that
folder. You can control the `subfolder` parameter and other file/path names if the base class gives you the option.

**Pushing to the Hub**

Pushing to the Hugging Face Hub is so much like the save method. The only difference is that the files are then uploaded
to the Hub after saving.


## Concept 4: Task-based Modeling & Training
Hezar is a practical library not a framework (it can be though!). That's why we decided to categorize models, trainers,
datasets, etc. under task names e.g, `speech_recognition`, `language_modeling`, etc. If you've worked with other
libraries, this might somewhat seem irrational, but trust us! For most users and usages this fits better!

Currently, all models, trainers and datasets are categorized by task name, but this does not mean that for every task,
there exists a model, trainer, dataset, etc.

## Concept 5: Integration with Other Tools
Re-inventing the wheel has no place in Hezar. It's strongly recommended that if something already exists somewhere, and
we want it, just copy and paste it into the code!<br>
In terms of backbone frameworks and libraries, we carefully R&D the present tools and choose the one that is the simplest
yet popular.

More specifically, here's a simple summary of the core modules in Hezar:
- **Models**:  Every model is a `hezar.models.Model` instance which is in fact, a PyTorch `nn.Module` wrapper with extra features for saving, loading, exporting, etc.
- **Datasets**: Every dataset is a `hezar.data.Dataset` instance which is a PyTorch Dataset implemented specifically for each task that can load the data files from the Hugging Face Hub.
- **Preprocessors**: All preprocessors are preferably backed by a robust library like Tokenizers, pillow, etc.
- **Embeddings**: All embeddings are developed on top of Gensim and can be easily loaded from the Hub and used in just 2 lines of code!
- **Trainer**: Trainer is the base class for training almost any model in Hezar or even your own custom models backed by Hezar. The Trainer comes with a lot of features and is also exportable to the Hub!
- **Metrics**: Metrics are also another configurable and portable modules backed by Scikit-learn, seqeval, etc. and can be easily used in the trainers!


## Concept 6: Our Inspirations
Hezar was built using the best practices we've learned from working with dozens of industry leading open source
software in the AI world. Our biggest inspirations are:

- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- [Fairseq](https://github.com/facebookresearch/fairseq) by Meta AI
- [Flair](https://github.com/flairNLP/flair) by FlairAI
- [ZenML](https://github.com/zenml-io/zenml)
- [Ludwig](https://github.com/ludwig-ai/ludwig) by Ludwig AI
- [UniLM](https://github.com/microsoft/unilm) by Microsoft
- [PyTorch Ignite](https://github.com/pytorch/ignite) by PyTorch
- [Lightning](https://github.com/Lightning-AI/lightning) by Lightning AI
- [Hazm](https://github.com/roshan-research/hazm) by Roshan
