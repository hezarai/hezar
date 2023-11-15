import os

from hezar.models import Model

TESTABLE_MODELS = {
    "automatic-speech-recognition": "hezarai/whisper-small-fa",
    "fill-mask": "hezarai/roberta-fa-mlm",
    "image-captioning": "hezarai/vit-roberta-fa-image-captioning-flickr30k",
    "ocr": "hezarai/crnn-base-fa-64x256",
    "text-classification": "hezarai/distilbert-fa-sentiment-digikala-snappfood",
    "text-generation": "hezarai/gpt2-base-fa",
    "sequence-labeling": "hezarai/bert-fa-pos-lscp-500k",
}

INVALID_OUTPUT_TYPE = "Model output must be a batch!"
INVALID_OUTPUT_SIZE = "Model output must be a list of size 1!"
INVALID_OUTPUT_FIELDS = "Invalid fields in the model outputs!"


def test_text_classification():
    inputs = ["هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"]
    model = Model.load(TESTABLE_MODELS["text-classification"])
    outputs = model.predict(inputs, top_k=2)
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], list)
    assert {k for x in outputs[0] for k in x.keys()} == {"label", "score"}


def test_sequence_labeling():
    inputs = ["شرکت هوش مصنوعی هزار"]
    model = Model.load(TESTABLE_MODELS["sequence-labeling"])
    outputs = model.predict(inputs, return_scores=True, return_offsets=True)
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], list)
    assert {k for el in outputs[0] for k in el.keys()} == {"label", "token", "start", "end", "score"}


def test_automatic_speech_recognition():
    dirname = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(dirname, "samples", "speech_example.mp3")
    model = Model.load(TESTABLE_MODELS["automatic-speech-recognition"])
    outputs = model.predict(inputs)
    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    assert set(outputs[0].keys()) == {"text", "chunks"}, INVALID_OUTPUT_FIELDS


def test_masked_language_modeling():
    inputs = ["سلام بچه ها حالتون <mask>"]
    model = Model.load(TESTABLE_MODELS["fill-mask"])
    outputs = model.predict(inputs)
    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    assert {k for x in outputs[0] for k in x.keys()} == {"token", "sequence", "token_id", "score"}, INVALID_OUTPUT_FIELDS


def test_text_generation():
    inputs = ["با پیشرفت اخیر هوش مصنوعی در سال های اخیر، "]
    model = Model.load(TESTABLE_MODELS["text-generation"])
    outputs = model.predict(inputs)
    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    assert set(outputs[0].keys()) == {"text"}, INVALID_OUTPUT_FIELDS


def test_ocr():
    dirname = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(dirname, "samples", "ocr_example.jpg")
    model = Model.load(TESTABLE_MODELS["ocr"])
    outputs = model.predict(inputs)
    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    assert set(outputs[0].keys()) == {"text"}, INVALID_OUTPUT_FIELDS


def test_image_captioning():
    dirname = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(dirname, "samples", "image_captioning_example.jpg")
    model = Model.load(TESTABLE_MODELS["image-captioning"])
    outputs = model.predict(inputs)
    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    assert set(outputs[0].keys()) == {"text"}, INVALID_OUTPUT_FIELDS
