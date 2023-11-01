# Preprocessors
A really important group of modules in Hezar is the preprocessors. Preprocessors are responsible for every single
processing of inputs from their rawest form to the point that they're ready to be fed to the model.

Preprocessors include all the tokenizers, feature extractors, normalizers, etc. and all of them are considered as a
preprocessor type.

## Loading preprocessors
Following the common pattern among all modules in Hezar, preprocessors also can be loaded in the same way.

**Loading with the corresponding module**<br>
You can load any preprocessor of any type with its base class like `Tokenizer`, `AudioFeatureExtractor`, etc.
```python
from hezar.preprocessors import Tokenizer, AudioFeatureExtractor, TextNormalizer

tokenizer = Tokenizer.load("hezarai/bert-base-fa")
normalizer = TextNormalizer.load("hezarai/roberta-base-fa")
feature_extractor = AudioFeatureExtractor.load("hezarai/whisper-small-fa")
...
```
**Loading with the Preprocessor module**<br>
Some models might need multiple types of preprocessors. For example encoder-decoder multimodal models like image captioning models
or even audio models need both feature extractor and text tokenizer or even a text normalizer. In order to load all
preprocessors in a path, you can use the `Preprocessor.load`. The output of this method depends on whether the path
contains single or multiple preprocessors.
- If path contains only one preprocessor the output is a preprocessor object of the right type.
- If path contains multiple preprocessors, the output is a `PreprocessorContainer` which is a dict-like object that holds
each preprocessor by its registry name.
```python
from hezar.preprocessors import Tokenizer

tokenizer = Tokenizer.load("hezarai/bert-base-fa")
print(tokenizer)
```
```
<hezar.preprocessors.tokenizers.wordpiece.WordPieceTokenizer object at 0x7f636d951e50>
```

```python
from hezar.preprocessors import Preprocessor

whisper_preprocessors = Preprocessor.load("hezarai/whisper-small-fa")
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

## Saving & Pushing to the Hub
Although preprocessor have their own type, they all implement the `load`, `save` and `push_to_hub` methods.
```python
from hezar.preprocessors import TextNormalizer, TextNormalizerConfig

normalizer = TextNormalizer(TextNormalizerConfig(nfkc=False))
normalizer.save("my-normalizer")
normalizer.push_to_hub("arxyzan/my-normalizer")
```
### Folder structure of the preprocessors
All preprocessors are saved under the `preprocessor` subfolder by default. Changing this behaviour is possible from all
three methods:
- `load(..., subfolder="SUBFOLDER")`
- `save(..., subfolder="SUBFOLDER")`
- `push_to_hub(..., subfolder="SUBFOLDER")`

The folder structure of the preprocessors for any save model (locally or in a repo) is something like below:
```
hezarai/whisper-small-fa
├── model_config.yaml
├── model.pt
└── preprocessor
    ├── feature_extractor_config.yaml
    ├── tokenizer_config.yaml
    └── tokenizer.json

```
