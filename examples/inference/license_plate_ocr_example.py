from hezar.models import Model


model = Model.load("hezarai/crnn-fa-license-plate-recognition-v2")
text = model.predict("../assets/license_plate_ocr_example.jpg")
print(text)
