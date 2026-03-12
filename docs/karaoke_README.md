# test3.py - 유명인사 연설 따라하기 (Speech Karaoke)

MediaPipe Holistic을 활용한 유명인사 연설 따라하기 핵심 로직 모듈입니다. 원본 영상과 사용자 웹캠을 비교하여 자세(Pose), 손동작(Hands), 얼굴 각도(Face)의 유사도를 실시간으로 분석합니다.

---

## 실행 방법

### 0. 자막 추출 (영상 대사 → JSON, Whisper 사용)

영상의 실제 대사를 음성 인식하여 자막 JSON을 생성합니다.

```bash
pip install faster-whisper
# ffmpeg 필요: brew install ffmpeg (macOS)

python extract_subtitles.py "Obama's 2004 DNC keynote speech.mp4"
# → Obama's 2004 DNC keynote speech_subs.json 생성

# 옵션: -m medium (더 정확), -o 출력경로
python extract_subtitles.py video.mp4 -m medium -o my_subs.json
```

### 1. 의존성 설치

```bash
pip install opencv-python mediapipe numpy
```

### 2. 기본 실행 (트레모 감지 예시)

```bash
python test3.py
```

실행 시 `_example_usage()` 함수가 호출되며, 트레모 감지 시뮬레이션 예시가 실행됩니다. (웹캠/영상 없이 동작)

### 3. Teacher Data 추출 (원본 영상 → JSON/CSV)

```python
from test3 import SpeechKaraokeTrainer

with SpeechKaraokeTrainer() as trainer:
    # 원본 영상에서 참조 데이터 추출 → reference_speech.json 저장
    ref_data = trainer.extract_reference_data(
        "reference_speech.mp4",
        output_format="json"  # 또는 "csv"
    )
```

### 4. 실시간 유사도 비교 (웹캠 + 참조 데이터)

```python
import cv2
from test3 import SpeechKaraokeTrainer
import json

# 참조 데이터 로드
with open("reference_speech.json", "r", encoding="utf-8") as f:
    ref = json.load(f)
ref_frames = ref["frames"]
fps = ref["fps"]

with SpeechKaraokeTrainer() as trainer:
    cap = cv2.VideoCapture(0)
    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = trainer._holistic.process(image_rgb)

        user_dict = trainer.normalize_user_frame(results)
        if user_dict and ref_frames:
            timestamp_ms = (frame_idx * 1000) / fps
            ref_frame = trainer.get_ref_frame_by_timestamp(ref_frames, timestamp_ms)
            similarity = trainer.calculate_pose_similarity(user_dict, ref_frame)
            print(f"유사도: {similarity * 100:.1f}%")

        frame_idx += 1
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
```

---

## 파일 구조

```
test3.py
├── 데이터 구조 및 상수
│   ├── FrameLandmarkData (dataclass)
│   ├── PoseLandmarkIndex (상수 클래스)
│   └── HAND_FINGERTIPS, HAND_MCP
│
├── SpeechKaraokeTrainer (핵심 클래스)
│   ├── Step 1: Teacher Data 추출
│   │   ├── extract_reference_data()      # 메인: 영상 → JSON/CSV
│   │   ├── _extract_single_frame_data()  # 단일 프레임 추출
│   │   ├── _normalize_vector()           # 벡터 정규화 (어깨 너비 기준)
│   │   ├── _compute_head_tilt_angles()   # Roll/Pitch/Yaw
│   │   ├── _compute_hand_open_ratio()   # 손 개폐 비율
│   │   └── _save_to_csv()               # CSV 저장
│   │
│   ├── Step 2: 실시간 유사도 계산
│   │   ├── calculate_pose_similarity()   # 메인: 코사인 유사도 계산
│   │   ├── _cosine_similarity_vectors()  # Body Pose 유사도
│   │   ├── _angle_similarity()           # Head Tilt 유사도
│   │   └── _hand_ratio_similarity()      # Hand Status 유사도
│   │
│   ├── Tremor 감지
│   │   └── detect_tremor()              # 떨림 수준 측정 (EMA 스무딩)
│   │
│   └── 헬퍼 메서드
│       ├── normalize_user_frame()       # 웹캠 → 비교용 dict
│       └── get_ref_frame_by_timestamp() # 타임스탬프 동기화
│
└── _example_usage()                      # 실행 시 호출되는 예시 함수
```

---

## 주요 클래스 및 메서드

| 구성요소 | 설명 |
|----------|------|
| **SpeechKaraokeTrainer** | 메인 클래스. MediaPipe Holistic 초기화 및 2단계 처리 로직 포함 |
| **extract_reference_data(video_path)** | 원본 mp4에서 프레임별 랜드마크/벡터 추출 → JSON 또는 CSV 저장 |
| **calculate_pose_similarity(user, ref)** | 사용자와 참조 데이터의 유사도 계산 (0~1, 100%=1.0) |
| **detect_tremor(landmark_history)** | 손/어깨 위치 이력의 분산으로 떨림 수준 측정 (0~1) |
| **normalize_user_frame(results)** | Holistic 결과를 `calculate_pose_similarity` 입력 형식으로 변환 |
| **get_ref_frame_by_timestamp(ref_data, ts)** | 타임스탬프에 가장 가까운 참조 프레임 반환 |

---

## 분석 대상 (Target Features)

| 항목 | 설명 | 구현 방식 |
|------|------|-----------|
| **Body Pose** | 어깨, 팔꿈치, 손목 각도 | 어깨→팔꿈치, 팔꿈치→손목 벡터의 코사인 유사도 |
| **Head Tilt** | 콧대와 어깨선 각도 | Roll(귀선), Pitch(끄덕임), Yaw(고개 돌림) |
| **Tremor** | 손/어깨 위치 변화 분산 | EMA 스무딩 + 분산(Variance) 측정, 노이즈 임계값 구분 |
| **Hand Status** | 손바닥 개폐 여부 | 손가락 끝↔밑마디 거리 → 0(주먹)~1(손 펴짐) |

---

## 출력 데이터 형식 (JSON)

`extract_reference_data`로 저장되는 JSON 구조:

```json
{
  "fps": 30.0,
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_ms": 0.0,
      "shoulder_elbow_wrist_vectors": {
        "left": [[...], [...]],
        "right": [[...], [...]]
      },
      "head_tilt_angles": { "roll": 0.0, "pitch": 0.0, "yaw": 0.0 },
      "hand_open_ratios": { "left": 0.5, "right": 0.5 },
      "shoulder_center": [0.5, 0.3],
      "shoulder_width": 0.2
    }
  ]
}
```

---

## 생성자 파라미터 (조정 가능)

```python
SpeechKaraokeTrainer(
    tremor_window_size=30,      # 떨림 분산 계산 프레임 수
    tremor_smooth_alpha=0.7,    # EMA 스무딩 강도 (낮을수록 노이즈 억제)
    tremor_noise_threshold=0.003  # 이 값 미만 = 노이즈로 간주
)
```

---

## 유사도 가중치 (calculate_pose_similarity)

기본값: `body_pose: 0.4`, `head_tilt: 0.3`, `hand_status: 0.3`

```python
similarity = trainer.calculate_pose_similarity(
    user_dict, ref_frame,
    weights={"body_pose": 0.5, "head_tilt": 0.25, "hand_status": 0.25}
)
```

---

## 참고

- **Test2_MP.py**와 동일한 `PoseLandmarkIndex`를 사용하여 연동이 용이합니다.
- 영상 처리 및 데이터 로직 중심이며, UI(오버레이 등)는 별도 구현이 필요합니다.
