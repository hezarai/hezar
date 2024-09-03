import os

import pytest

from hezar.models import Model
from hezar.utils import clean_cache

CI_MODE = os.environ.get("CI_MODE", "FALSE")

TEST_CASES = {
    "mask-filling": {
        "model": "hezarai/roberta-fa-mask-filling",
        "input": "سلام بچه ها حالتون <mask>",
    },
    "sequence-labeling": {
        "model": "hezarai/bert-fa-pos-lscp-500k",
        "input": "شرکت هوش مصنوعی هزار",
    },
    "text-classification": {
        "model": "hezarai/distilbert-fa-sentiment-dksf",
        "input": "هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی",
    },
    "text-generation": {"model": "hezarai/gpt2-base-fa", "input": "با پیشرفت اخیر هوش مصنوعی در سال های اخیر، "},
}

OUTPUT_MISMATCH = "Pickle and Safetensor model outputs are not a match!"


@pytest.mark.parametrize("task", TEST_CASES.keys())
def test_safetensors(task):
    model = TEST_CASES[task]["model"]
    sample = TEST_CASES[task]["input"]

    pickled_model = Model.load(model)
    pickled_outputs = pickled_model.predict(sample)
    pickled_model.save("safetensors_model", safe_serialization=True)

    safe_model = Model.load("safetensors_model", load_safetensors=True)
    safe_outputs = safe_model.predict(sample)

    assert pickled_outputs == safe_outputs, OUTPUT_MISMATCH

    if CI_MODE == "TRUE":
        clean_cache(delay=1)
