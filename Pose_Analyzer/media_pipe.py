#pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# MediaPipe 구성요소 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 웹캠 캡처 객체 생성 (일반적으로 0번이 내장/기본 웹캠)
cap = cv2.VideoCapture(0)

# Holistic 모델 로드
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            continue

        # 1. 이미지 전처리 (OpenCV는 BGR, MediaPipe는 RGB를 사용)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 모델 추론 (얼굴, 몸통, 손 랜드마크 추출)
        results = holistic.process(image)

        # 3. 결과 렌더링을 위해 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 4. 랜드마크 그리기
        # 얼굴 메쉬 그리기
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        
        # 몸(Pose) 뼈대 그리기
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # 양손 그리기
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 화면 출력 및 종료 키(q) 설정
        cv2.imshow('Interview Feedback - Posture Detection', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()