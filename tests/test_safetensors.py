from hezar.models import Model


def test_safetensors():
    example = ["هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"]

    pickled_model = Model.load("hezarai/bert-fa-sentiment-dksf")
    pickled_outputs = pickled_model.predict(example)

    pickled_model.save("safetensors_model", safe_serialization=True)

    safe_model = Model.load("safetensors_model", load_safetensors=True)
    safe_outputs = safe_model.predict(example)

    assert pickled_outputs == safe_outputs
