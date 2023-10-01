from hezar import Model


path = "roberta-base-fa"
text = ["سلام بچه ها حالتون <mask>"]
model = Model.load(path)
outputs = model.predict(text)
print(outputs)
