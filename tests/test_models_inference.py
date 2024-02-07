import os

import pytest

from hezar.builders import build_model
from hezar.models import ModelConfig
from hezar.preprocessors import Preprocessor
from hezar.utils import clean_cache

CI_MODE = os.environ.get("CI_MODE", "FALSE")


TESTABLE_MODELS = {
    "speech-recognition": {
        "path": "hezarai/whisper-small-fa",
        "inputs": {"type": "file", "value": "samples/speech_example.mp3"},
        "predict_kwargs": {},
        "output_type_within_batch": dict,
        "required_output_keys": {"text", "chunks"}
    },
    "mask-filling": {
        "path": "hezarai/roberta-fa-mask-filling",
        "inputs": {"type": "text", "value": "سلام بچه ها حالتون <mask>"},
        "predict_kwargs": {"top_k": 3},
        "output_type_within_batch": list,
        "required_output_keys": {"token", "sequence", "token_id", "score"}
    },
    "image-captioning": {
        "path": "hezarai/vit-roberta-fa-image-captioning-flickr30k",
        "inputs": {"type": "file", "value": "samples/image_captioning_example.jpg"},
        "predict_kwargs": {},
        "output_type_within_batch": dict,
        "required_output_keys": {"text", "score"}
    },
    "ocr": {
        "path": "hezarai/crnn-fa-printed-96-long",
        "inputs": {"type": "file", "value": "samples/ocr_example.jpg"},
        "predict_kwargs": {"return_scores": True},
        "output_type_within_batch": dict,
        "required_output_keys": {"text", "score"}
    },
    "text-classification": {
        "path": "hezarai/distilbert-fa-sentiment-dksf",
        "inputs": {"type": "text", "value": "هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"},
        "predict_kwargs": {"top_k": 2},
        "output_type_within_batch": list,
        "required_output_keys": {"label", "score"}
    },
    "text-generation": {
        "path": "hezarai/gpt2-base-fa",
        "inputs": {"type": "text", "value": "با پیشرفت اخیر هوش مصنوعی در سال های اخیر، "},
        "predict_kwargs": {},
        "output_type_within_batch": dict,
        "required_output_keys": {"text"}
    },
    "sequence-labeling": {
        "path": "hezarai/bert-fa-pos-lscp-500k",
        "inputs": {"type": "text", "value": "شرکت هوش مصنوعی هزار"},
        "predict_kwargs": {"return_offsets": True, "return_scores": True},
        "output_type_within_batch": list,
        "required_output_keys": {"label", "token", "start", "end", "score"}
    }
}

INVALID_OUTPUT_TYPE = "Model output must be a batch!"
INVALID_OUTPUT_SIZE = "Model output must be a list of size 1!"
INVALID_OUTPUT_FIELDS = "Invalid fields in the model outputs!"


@pytest.mark.parametrize("task", TESTABLE_MODELS.keys())
def test_model_inference(task):
    model_params = TESTABLE_MODELS[task]
    path = model_params["path"]
    predict_kwargs = model_params["predict_kwargs"]
    output_type_within_batch = model_params["output_type_within_batch"]
    required_output_keys = model_params["required_output_keys"]

    if model_params["inputs"]["type"] == "file":
        dirname = os.path.dirname(os.path.abspath(__file__))
        inputs = os.path.join(dirname, model_params["inputs"]["value"])
    else:
        inputs = model_params["inputs"]["value"]

    model_config = ModelConfig.load(path)
    model = build_model(model_config.name, config=model_config)
    model.preprocessor = Preprocessor.load(path)

    outputs = model.predict(inputs, **predict_kwargs)

    assert isinstance(outputs, list), INVALID_OUTPUT_TYPE
    assert len(outputs) == 1, INVALID_OUTPUT_SIZE
    if output_type_within_batch == list:
        assert {k for el in outputs[0] for k in el.keys()} == required_output_keys
    elif output_type_within_batch == dict:
        assert set(outputs[0].keys()) == required_output_keys, INVALID_OUTPUT_FIELDS

    if CI_MODE == "TRUE":
        clean_cache(delay=1)
