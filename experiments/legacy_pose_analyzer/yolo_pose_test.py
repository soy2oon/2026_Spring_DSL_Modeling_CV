# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# YOLOv8-Pose 모델 로드 (n=nano, s=small, m=medium, l=large, x=xlarge)
# nano가 가장 가볍고 빠름, 실시간 웹캠에 적합
model = YOLO("yolov8n-pose.pt")

# 웹캠 캡처
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        continue

    # 포즈 추론 (stream=True로 메모리 효율적 처리)
    results = model(frame, verbose=False)

    # 결과 시각화 (키포인트 + 스켈레톤 자동 그리기)
    annotated_frame = results[0].plot()

    cv2.imshow("Interview Feedback - YOLOv8 Pose", annotated_frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
