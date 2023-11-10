from hezar.models import Model


model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")  # CRNN
text = model.predict("../assets/license_plate_ocr_example.jpg")
print(text)
