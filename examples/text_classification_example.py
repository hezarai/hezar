from hezar.models.text_classification.distilbert import DistilBertTextClassification

model = DistilBertTextClassification.load('hezar-ai/distilbert-fa-zwnj-base-sentiment')
text = ['یه تست خیلی خفن']
outputs = model.predict(text)
print(outputs)
