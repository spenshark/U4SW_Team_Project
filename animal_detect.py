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

original_height, original_width, original_channal = img.shape

resized_image = cv2.resize(img, (target_width, target_height))

# 이미지로부터 blob 얻기
blob = cv2.dnn.blobFromImage(resized_image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
print('blob shape:', blob.shape)

# coco 객체 이름 읽기
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# yolo 모델 로드
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# 출력 레이어 설정
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('output layers:', output_layers)

# 겍체 탐지
net.setInput(blob)
outs = net.forward(output_layers)

# 바운딩 박스 및 신뢰도 가져오기
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object

    for detection in out: # for each bounding box

        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > 0.5:
            # 신뢰도 기준으로 필터링 및 바운딩 박스 좌표 계산
            center_x = int(detection[0] * original_width)
            center_y = int(detection[1] * original_height)
            w = int(detection[2] * original_width)
            h = int(detection[3] * original_height)

            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)


# 판단한 클래스 별로 OpenCV로 표현
class_ids = []
confidence_scores = []
boxes = []
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

cv2.imshow("Resized Image", resized_image)
cv2.imshow("Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()