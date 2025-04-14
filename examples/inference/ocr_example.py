from hezar.models import Model

model = Model.load("hezarai/crnn-base-fa-v2")
text = model.predict("../assets/ocr_example.jpg")
print(text)
