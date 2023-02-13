from transformers import AutoTokenizer

from hezar.models import Model
path = 'hezar-ai/distilbert-fa-sentiment-v1'
model = Model.load(path)
model.tokenizer = AutoTokenizer.from_pretrained(path)
text = ['یه تست خیلی خفن']
outputs = model.predict(text)
print(outputs)
