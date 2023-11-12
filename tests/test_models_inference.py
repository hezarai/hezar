from unittest import TestCase

from hezar.models import Model

TESTABLE_MODELS = {
    "automatic-speech-recognition": "hezarai/whisper-small-fa",
    "fill-mask": "hezarai/roberta-fa-mlm",
    "image-to-text": "hezarai/vit-roberta-fa-image-captioning-flickr30k",
    "text-classification": "hezarai/distilbert-fa-sentiment-digikala-snappfood",
    "text2text-generation": "hezarai/gpt2-base-fa",
    "token-classification": "hezarai/bert-fa-pos-lscp-500k",
}

# TODO
