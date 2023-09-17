from hezar import Model

model = Model.load("hezarai/trocr-fa-v1")
text = model.predict(["assets/ocr_example.jpg"])
print(text)
