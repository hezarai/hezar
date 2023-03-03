from hezar.preprocessors import Tokenizer

from hezar.builders import build_model
from hezar.models import Model

path = "hezar-ai/distilbert-fa"
tokenizer = Tokenizer.load(path)
text = tokenizer(["hello guys"], return_tensors="pt")

model = build_model("distilbert_lm")
outputs = model.predict(text, output_attentions=True, output_hidden_states=True)
print(outputs)

model = Model.load(path, save_to_cache=True)
outputs = model.predict(text, output_attentions=True, output_hidden_states=True)
print(outputs)
