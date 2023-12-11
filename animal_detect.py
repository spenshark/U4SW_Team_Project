import cv2
import numpy as np

# 사진 넣기
# display resolution: FHD
target_width = 1920	
target_height = 1080

image_name = input('enter your image name: ')

image_path = './image/'+image_name+'.jpg'

img = cv2.imread(image_path)

if img is None:
	print("\nEroor: File not found or unable to load.")
	exit()

original_height, original_width = img.shape[:2]

resized_image = cv2.resize(img, (target_width, target_height))

cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 판단한 클래스 별로 OpenCV로 표현
class_ids = []
confidence_scores = []
boxes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

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
