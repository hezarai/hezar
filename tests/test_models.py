from typing import *
from omegaconf import DictConfig

from hezar import models_registry, load_model
from hezar.models import Model


def test_build_distilbert_text_classification():
    model_name = 'distilbert_text_classification'
    model = models_registry[model_name]['model_class'](models_registry[model_name]['model_config'](num_labels=3))
    print(model)


def test_distilbert_save_model():
    name = 'distilbert_text_classification'
    inner_model_config = DictConfig(
        {'pretrained_model_name_or_path': 'hezar-ai/distilbert-fa-zwnj-base', 'num_labels': 10})
    model: Type[Model] = load_model(name, mode='training', inner_model_config=inner_model_config)
    model.save_pretrained(path='../examples/test_save')
    print(model)


def test_distilbert_text_classification_prediction():
    model_name = 'distilbert_text_classification'
    model = models_registry[model_name]['model_class'].from_pretrained('test')
    text = 'hello from Hezar!'
    print(model.predict(text))


if __name__ == '__main__':
    # test_distilbert_save_model()
    test_build_distilbert_text_classification()
    # test_distilbert_text_classification_prediction()
