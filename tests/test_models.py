from hezar.models.text_classification import DistilBertTextClassification, RobertaTextClassification


def test_build_distilbert_text_classification():
    model = DistilBertTextClassification.from_hub('test')
    print(model)


def test_distilbert_text_classification_prediction():
    model = DistilBertTextClassification.from_hub('test')
    text = 'hello from Hezar!'
    print(model.predict(text))


def test_build_roberta_text_classification():
    model = RobertaTextClassification.from_hub('test')
    print(model)


def test_roberta_text_classification_prediction():
    model = RobertaTextClassification.from_hub('test')
    text = 'hello from Hezar!'
    print(model.predict(text))


if __name__ == '__main__':
    test_build_distilbert_text_classification()
    test_distilbert_text_classification_prediction()
