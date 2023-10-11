# -*- coding: utf-8 -*-
from hezar import Model

hub_path = "hezarai/distilbert-fa-sentiment-dksf"
model = Model.load(hub_path, device="cpu")
inputs = ["کتابخانه هزار، بهترین کتابخانه هوش مصنوعیه"]
model_outputs = model.predict(inputs)
print(model_outputs)
