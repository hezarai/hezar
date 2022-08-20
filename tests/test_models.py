from hezar import models_registry


def test_build_distilbert_text_classification():
    model_name = 'distilbert_text_classification'
    model = models_registry[model_name]['model_class'].from_hub('test')
    print(model)


def test_distilbert_text_classification_prediction():
    model_name = 'distilbert_text_classification'
    model = models_registry[model_name]['model_class'].from_hub('test')
    text = 'hello from Hezar!'
    print(model.predict(text))


def test_build_roberta_text_classification():
    model_name = 'roberta_text_classification'
    model = models_registry[model_name]['model_class'].from_hub('test')
    print(model)


def test_roberta_text_classification_prediction():
    model_name = 'roberta_text_classification'
    model = models_registry[model_name]['model_class'].from_hub('test')
    text = 'hello from Hezar!'
    print(model.predict(text))


if __name__ == '__main__':
    test_build_distilbert_text_classification()
    test_distilbert_text_classification_prediction()
