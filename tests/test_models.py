from transformers import AutoTokenizer

from hezar.builders import build_model
from hezar.models import Model


def test_build_distilbert_text_classification():
    model_name = "distilbert_text_classification"
    model = build_model(model_name, num_labels=10)
    print(model)


def test_model_save_load():
    model_name = "distilbert_text_classification"
    model1 = build_model(model_name, num_labels=10)
    model1.save("saved/test")
    model2 = Model.load("saved/test")
    print(model2)


def test_load():
    from hezar.configs import ModelConfig

    config = ModelConfig.load("hezar-ai/distilbert-fa-sentiment-v1")
    print(config)


def test_distilbert_text_classification_prediction():
    model_name = "distilbert_text_classification"
    model = build_model(model_name, num_labels=10)
    model.push_to_hub("hezar-ai/distilbert-fa-sentiment-v1")
    model.tokenizer = AutoTokenizer.from_pretrained("hezar-ai/distilbert-fa-sentiment-v1")
    text = "hello from Hezar!"
    outputs = model.predict(text)
    print(outputs)


def test_mlp():
    model_name = "mlp"
    model = build_model(model_name)
    outputs = model.predict([1, 2, 3, 4])
    print(outputs)
    model.push_to_hub("hezar-ai/mlp-test")


if __name__ == "__main__":
    test_load()
    # test_distilbert_save_model()
    # test_build_distilbert_text_classification()
    # test_distilbert_text_classification_prediction()
    # test_model_save_load()
    # test_mlp()
