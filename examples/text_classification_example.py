from hezar import build_model, Model, Tokenizer

path = "hezar-ai/bert-fa-sentiment-digikala-snappfood"
model = Model.load(path)
# model = build_model('distilbert_text_classification', id2label={0: 'negative', 1: 'positive'})
tokenizer = Tokenizer.load(path)
inputs = tokenizer(["یه تست خیلی خفن"], return_tensors="pt")
outputs = model.predict(inputs)
print(outputs)
