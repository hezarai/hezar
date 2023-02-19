from transformers import AutoTokenizer, DistilBertModel

from hezar.models import build_model, Model

path = 'hezar-ai/distilbert-fa'
tokenizer = AutoTokenizer.from_pretrained(path)
text = ['hello guys']

model = build_model('distilbert_lm')
model.tokenizer = tokenizer
outputs = model.predict(text, output_attentions=True, output_hidden_states=True)
print(outputs)

model = Model.load(path, save_to_cache=True)
model.tokenizer = tokenizer
outputs = model.predict(text, output_attentions=True, output_hidden_states=True)
print(outputs)
