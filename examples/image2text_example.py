from hezar import Model

model = Model.load("hezarai/trocr-fa-v1")
text = model.predict(["image.jpg"])
print(text)
