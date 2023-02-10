from transformers import AutoTokenizer

from hezar.models import build_model, Model


def test_build_distilbert_text_classification():
    model_name = 'distilbert_text_classification'
    model = build_model(model_name, num_labels=10)
    print(model)


def test_model_save_load():
    model_name = 'distilbert_text_classification'
    model1 = build_model(model_name, num_labels=10)
    model1.save('saved/test')
    model2 = Model.load('saved/test')
    print(model2)


def test_load():
    from hezar.configs import ModelConfig
    config = ModelConfig.load('hezar-ai/distilbert-fa-sentiment-v1')
    print(config)


def test_distilbert_text_classification_prediction():
    model_name = 'distilbert_text_classification'
    model = build_model(model_name, num_labels=10)
    model.tokenizer = AutoTokenizer.from_pretrained('hezar-ai/distilbert-fa-sentiment-v1')
    text = 'hello from Hezar!'
    outputs = model.predict(text)
    print(outputs)


if __name__ == '__main__':
    # test_load()
    # test_distilbert_save_model()
    # test_build_distilbert_text_classification()
    test_distilbert_text_classification_prediction()
    # test_model_save_load()
