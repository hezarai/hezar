from hezar.models import Model

model = Model.load('hezar-ai/distilbert-fa-sentiment-v1')
text = ['یه تست خیلی خفن']
outputs = model.predict(text)
print(outputs)
