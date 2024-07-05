from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image


model = Model.load("craft", device="cuda")
image = load_image("../assets/text_detection_example.jpg")
outputs = model.predict(image)
result_image = draw_boxes(image, outputs[0]["boxes"])
show_image(result_image, "text_detected")
