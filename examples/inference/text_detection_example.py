import cv2
from hezar.models import Model
from hezar.utils import draw_boxes


model = Model.load("craft", device="cuda")
outputs = model.predict("../assets/text_detection_example.png")
image = cv2.imread("../assets/text_detection_example.png")
result_image = draw_boxes(image, outputs[0]["boxes"])
cv2.imwrite("detected.png", result_image)
