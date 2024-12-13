# Install dependencies
# pip install ultralytics

from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

model_path = ""  # path to the weights file
model = YOLO(model_path)

image_path = ""  # path to the image
while True:
    results = model.predict(source=image_path, imgsz=640, conf=0.25)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            print(f'area :{width * height}')

