from hezar.models import Model

model = Model.load("hezarai/crnn-fa-printed-96-long")
text = model.predict("../assets/ocr_example.jpg")
print(text)
