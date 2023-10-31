from hezar.models import Model


path = "hezarai/roberta-fa-mlm"
text = ["سلام بچه ها حالتون <mask>"]
model = Model.load(path)
outputs = model.predict(text)
print(outputs)
