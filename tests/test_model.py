from hezar import DistilBertTextClassification


def test_model():
    model = DistilBertTextClassification.from_pretrained('test')


if __name__ == '__main__':
    test_model()
