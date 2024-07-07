import pprint
from hezar.models import Model
from hezar.utils import load_image, crop_boxes, save_image


detector_model = Model.load("hezarai/CRAFT", device="cuda")
recognition_model = Model.load("hezarai/crnn-fa-printed-96-long", device="cuda")

image = load_image("../assets/text_detection_example.png")

detection_outputs = detector_model.predict(image)

boxes = detection_outputs[0]["boxes"]
text_images = crop_boxes(image, boxes, padding=(0, 4, 0, 4))  # Add up/down padding for better accuracy

results = []
for text_image, box in zip(text_images, boxes):
    text = recognition_model.predict(text_image)[0]["text"]
    save_image(text_image, f"outputs/{text}.png")
    results.append({"box": box, "text": text})

pprint.pprint(results)
