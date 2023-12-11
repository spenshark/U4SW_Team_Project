import cv2
import numpy as np

class_ids = []
confidence_scores = []
boxes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# load pre-trained yolo model or whatever from configuration and weight files
# classify the image

indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(f"class {label} detected at {x}, {y}, {w}, {h}")
        color = colors[i]
        if label == "dog":
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

cv2.imshow("Objects", img)
