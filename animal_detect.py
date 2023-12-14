import cv2
import numpy as np
import os
import sys

# 상수 정의
TARGET_WIDTH = 1920	
TARGET_HEIGHT = 1080
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5

# 이미지 경로
image_name = input('enter your image name: ')
image_path = f'./image/{image_name}.jpg'

# 이미지 파일 확인
if not os.path.exists(image_path):
    sys.exit(f"Error: Unable to find the image at {image_path}")

# 이미지 로드 및 크기 조정
img = cv2.imread(image_path)
resized_image = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
original_height, original_width, original_channel = img.shape

# blob 얻기
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

# 객체 탐지
net.setInput(blob)
outs = net.forward(output_layers)

# 바운딩 박스 및 신뢰도 가져오기
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object
    for i, detection in enumerate(out): # for each bounding box
        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > CONFIDENCE_THRESHOLD:
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

# 판단한 클래스 별로 OpenCV 표현
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

# 탐지한 이미지를 저장할 폴더 생성
output_folder = "detected_objects"
os.makedirs(output_folder, exist_ok=True)

def save_and_display_objects(image, boxes, indices, classes, confidence_scores, colors, output_folder):
    cnt = 0
    for i in indices:
        x, y, w, h = boxes[i]
        roi = image[y:y+h, x:x+w]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cnt += 1
        
        # 탐지한 이미지를 새 이미지로 저장
        output_path = f"{output_folder}/{label}_{cnt}.png"
        print(f"Object {label}_{cnt} saved at: {output_path}")
        cv2.imwrite(output_path, roi)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
        cv2.putText(image, label, (x, y - 10), font, 1, color, 2)

        # 객체의 정보를 파일에 기록
        log_file_path = "object_detection_log.txt"
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Object {cnt}: Class: {label}, Confidence: {confidence_scores[i]:.2f}, Coordinates: ({x}, {y}, {w}, {h})\n")

# 객체 탐지 및 표시
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, NMS_THRESHOLD, 0.4)
save_and_display_objects(img, boxes, indices, classes, confidence_scores, colors, output_folder)

cv2.imshow("Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
