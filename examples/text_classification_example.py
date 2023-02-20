from hezar.models import Model
from hezar.preprocessors.tokenizer import Tokenizer

path = 'hezar-ai/distilbert-fa-sentiment-v1'
model = Model.load(path)
tokenizer = Tokenizer.load(path)
inputs = tokenizer(['یه تست خیلی خفن'], return_tensors='pt')
outputs = model.predict(inputs)
print(outputs)
