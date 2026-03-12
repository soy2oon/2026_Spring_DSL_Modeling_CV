# 면접 피드백 시스템 - 상체 동작 인식 (Pose Recognition)
## 📁 프로젝트 구조

```
CV_Modeling/
├── media_pipe.py      # MediaPipe Holistic 백본 (얼굴, 손, 포즈 랜드마크 시각화)
├── Test2_MP.py        # 메인 모듈 - 상체 동작 인식 + 피드백 파이프라인
├── pose_learn.py      # 포즈 학습 (주먹 자세 등 사용자 정의 동작 학습)
├── punch_pose_samples.json  # 학습된 샘플 저장 (pose_learn 실행 시 생성)
└── README.md
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
cd CV_Modeling
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install opencv-python mediapipe numpy Pillow
```

### 2. 메인 피드백 실행

```bash
python Test2_MP.py
# 종료: q 키
```

### 3. 주먹 자세 학습 (선택)

```bash
python pose_learn.py
# s 키: 현재 포즈를 주먹 샘플로 저장
# q 키: 종료 및 punch_pose_samples.json 저장
```

---

## 🏗 아키텍처 (DDD 구조)

팀 협업을 위해 **Domain-Driven Design** 형식으로 설계되었습니다.

| 레이어 | 역할 | 주요 클래스/함수 |
|--------|------|------------------|
| **Domain** | 비즈니스 규칙, 도메인 모델 | `AlertType`, `AlertMessage`, `PoseMetrics`, `PoseLandmarkIndex`, `PoseAnalyzer` |
| **Application** | 유스케이스 | `AlertChecker` |
| **Infrastructure** | 외부 의존성 (MediaPipe) | `HolisticBackbone`, `MediaPipeHolisticBackbone`, `HolisticResult` |
| **Interface** | 알림 표시, 메인 루프 | `AlertPresenter`, `OverlayAlertPresenter`, `run_pose_feedback_pipeline()` |

---

## 📋 전체 동작 방식

### 데이터 흐름

```
웹캠 프레임 (BGR)
       ↓
   RGB 변환
       ↓
MediaPipe Holistic process()
       ↓
pose_landmarks, face_landmarks, left/right_hand_landmarks
       ↓
PoseAnalyzer.analyze()  →  PoseMetrics
       ↓
AlertChecker.check_alerts()  →  list[AlertMessage]
       ↓
OverlayAlertPresenter.update_and_show()  →  화면에 알림 오버레이
```

### 1. MediaPipe Holistic

- **입력**: RGB 이미지
- **출력**: 33개 포즈 랜드마크, 468개 얼굴 랜드마크, 양손 각 21개
- **사용**: `media_pipe.py`의 기본 루프 또는 `Test2_MP.py`에서 직접 호출

### 2. PoseAnalyzer (포즈 분석)

각 프레임마다 다음을 계산합니다.

| 지표 | 계산 방식 | 용도 |
|------|-----------|------|
| **body_tilt_angle** | 코~어깨중심 벡터가 화면 수직선과 이루는 각도 | 몸 기울기 |
| **head_tilt_angle** | 좌우 귀를 잇는 선의 수평선 대비 각도 | 머리 삐딱함 |
| **neck_head_tilt_angle** | 어깨선 대비 귀선 각도 차이 | 상체는 똑바른데 머리만 기울어진 경우 |
| **tremor_level** | 어깨 중심 위치의 이동 분산 (EMA 스무딩 후) | 몸 떨림 |
| **is_punch_gesture** | 규칙 기반 + 학습 샘플 유사도 | 주먹 이스터에그 |

### 3. AlertChecker (알림 판단)

`PoseMetrics`와 임계값을 비교해 알림 리스트를 반환합니다.

| 조건 | 알림 메시지 |
|------|-------------|
| body_tilt_angle ≥ 10° | "몸을 바로 세워주세요" |
| head_tilt ≥ 25° 또는 neck_head_tilt ≥ 20° | "머리를 바로 세워주세요" |
| tremor_level ≥ 0.85 (워밍업 후) | "몸의 떨림이 심합니다" |
| is_punch_gesture | "면접관을 가격해서는 안됩니다!" |

### 4. OverlayAlertPresenter (알림 표시)

- 알림 발생 시 **약 3초** 동안 빨간 배경 + 한글 메시지 표시
- `cv2.putText`는 한글 미지원 → **PIL(Pillow)**로 렌더링 후 합성

---

## 📐 상세 알고리즘

### 몸 기울기 (body_tilt)

- **화면 중심 수직선**을 기준으로 상체가 틀어진 정도
- 몸 중심축: 어깨 중심 → 코
- `angle = arctan2(|dx|, |dy|)` (dx, dy: 어깨중심→코 벡터)

### 머리/목 기울기 (head_tilt, neck_head_tilt)

- **head_tilt**: 좌우 귀를 잇는 선이 수평선과 이루는 각도
- **neck_head_tilt**: 어깨선과 귀선 각도의 차이 → 상체는 곧은데 머리만 기울어진 경우 감지

### 몸 떨림 (tremor)

1. 매 프레임 어깨 중심 `(cx, cy)` 계산
2. **EMA 스무딩**: `smoothed = α × smoothed_prev + (1-α) × current` (α=0.35)
3. 스무딩된 위치 20프레임의 **분산** 계산
4. `level = min(1, variance / 0.002)` 로 0~1 정규화
5. 시작 후 90프레임(~3초) 워밍업 구간은 알림 무시

### 주먹 감지 (punch)

**규칙 기반:**

1. 팔이 거의 직선 (wrist–elbow–shoulder 각도 > 150°)
2. 손이 주먹 상태 (손가락 끝↔밑마디 평균 거리 < 0.06)
3. 손목이 어깨보다 카메라에 가까움 (z 값)

**학습 기반 (pose_learn 사용 시):**

- `punch_pose_samples.json`의 샘플과 pose 특징 벡터 유사도 0.65 이상이면 주먹으로 판정

---

## ⚙️ 주요 파라미터 (PoseAnalyzer)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| BODY_TILT_THRESHOLD_DEG | 10.0 | 몸 기울기 알림 각도 (°) |
| HEAD_TILT_THRESHOLD_DEG | 25.0 | 머리 삐딱함 알림 각도 |
| NECK_HEAD_TILT_THRESHOLD_DEG | 20.0 | 목 기울기 알림 각도 |
| TREMOR_ALERT_THRESHOLD | 0.85 | 떨림 알림 기준 (0~1) |
| TREMOR_WARMUP_FRAMES | 90 | 시작 후 알림 무시 프레임 수 |
| TREMOR_SMOOTH_ALPHA | 0.35 | EMA 스무딩 강도 (낮을수록 jitter 억제) |
| PUNCH_FIST_CLOSED_THRESHOLD | 0.06 | 주먹 감지 손가락 거리 임계값 |

---

## 👥 팀 협업 가이드

### 파트 분리

- **Pose 파트 (현재)**: `PoseAnalyzer`, `AlertChecker` → 몸/머리/떨림/주먹
- **Face 파트**: `face_landmarks` → 표정, 시선 분석
- **Hand 파트**: `hand_landmarks` → 제스처 (주먹 제외)

### 다른 백본 사용

```python
from Test2_MP import create_pose_feedback_service, HolisticBackbone

class MyBackbone(HolisticBackbone):
    def process(self, image_rgb):
        # 자체 추론 로직
        return HolisticResult(pose_landmarks=..., face_landmarks=..., ...)

analyzer, checker = create_pose_feedback_service(MyBackbone())
```

### 알림 타입 추가

`AlertType`에 추가 후 `AlertMessage` 팩토리 메서드, `AlertChecker.check_alerts()` 로직을 확장하면 됩니다.

---

## 📦 의존성

- **opencv-python**: 웹캠 캡처, 화면 표시
- **mediapipe**: Holistic (포즈, 얼굴, 손)
- **numpy**: 수치 연산
- **Pillow**: 한글 텍스트 렌더링

---

---

## 📖 파일별 코드 설명

### Test2_MP.py

**Domain Layer (도메인 레이어)**

- `AlertType`: 알림 종류 enum (BODY_TILT, HEAD_TILT, BODY_TREMOR, PUNCH_GESTURE)
- `AlertMessage`: 알림 메시지 Value Object, `body_tilt()`, `head_tilt()` 등 팩토리 메서드 제공
- `PoseMetrics`: 분석 결과 (각도, tremor_level, is_punch_gesture 등)
- `PoseLandmarkIndex`: MediaPipe pose 랜드마크 인덱스 (NOSE=0, LEFT_SHOULDER=11 등)
- `PoseAnalyzer`: 핵심 분석 로직
  - `analyze()`: Holistic 결과 → PoseMetrics
  - `_compute_body_tilt()`: 코~어깨 벡터 각도
  - `_compute_head_tilt()`: 귀선 각도
  - `_compute_neck_head_tilt()`: 어깨선 vs 귀선 각도 차이
  - `_compute_tremor()`: EMA 스무딩 후 분산
  - `_detect_punch_gesture()`: 규칙/학습 기반 주먹 감지

**Application Layer (애플리케이션 레이어)**

- `AlertChecker`: PoseMetrics와 임계값 비교 → 알림 리스트 반환

**Infrastructure Layer (인프라 레이어)**

- `HolisticBackbone`: 백본 추상 인터페이스
- `MediaPipeHolisticBackbone`: MediaPipe Holistic 구현체
- `HolisticResult`: 추론 결과 데이터 클래스

**Interface Layer (인터페이스 레이어)**

- `_put_text_korean()`: PIL로 한글 렌더링 후 OpenCV 프레임에 합성
- `OverlayAlertPresenter`: 알림 3초 유지, 여러 알림 수직 배치
- `run_pose_feedback_pipeline()`: 웹캠 → Holistic → 분석 → 알림 → 화면 출력 루프

---

### pose_learn.py

- `landmarks_to_feature()`: pose + hand 랜드마크 → 특징 dict (어깨·팔꿈치·손목 x,y,z + 손 21점)
- `load_samples()` / `save_samples()`: `punch_pose_samples.json` 읽기/쓰기
- `run_learn_punch()`: 웹캠 루프, `s`로 샘플 저장, `q`로 종료 및 저장
- `compute_similarity()`: 두 특징 벡터 L2 거리 → 가우시안 유사도 (0~1)
- `is_punch_from_learned()`: 현재 포즈와 학습 샘플 유사도 ≥ threshold면 True

---

### media_pipe.py

MediaPipe Holistic 기본 데모. BGR→RGB 변환 후 `holistic.process()`, 결과로 얼굴/포즈/손 랜드마크를 그립니다. `Test2_MP.py`는 이와 유사하게 Holistic을 사용하되, 분석·알림 로직을 추가합니다.

---

## 📄 라이선스

이 프로젝트는 MediaPipe를 사용합니다. MediaPipe는 Apache 2.0 라이선스 하에 제공됩니다.
