"""
면접 피드백 시스템 - 상체 동작 인식 모듈 (Pose Recognition)

협업 시 파트 분리:
- 다른 팀원: face_landmarks 기반 표정/시선 → domain/face 모듈
- 다른 팀원: hand_landmarks 기반 제스처 → domain/gesture 모듈 (주먹은 pose 파트)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# 한글 렌더링용 (cv2.putText는 한글 미지원)
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# pose_learn.py 연동 (주먹 자세 학습)
try:
    from .pose_learn import load_samples, is_punch_from_learned
except ImportError:
    load_samples = None
    is_punch_from_learned = None

LEARNED_PUNCH_PATH = Path(__file__).resolve().parent / "punch_pose_samples.json"


# =============================================================================
# DOMAIN LAYER - 도메인 모델 및 비즈니스 규칙
# =============================================================================


class AlertType(Enum):
    """알림 유형 - 팀 전체에서 공유 가능한 도메인 상수"""

    BODY_TILT = "body_tilt"  # 몸 기울기
    HEAD_TILT = "head_tilt"  # 머리/목 기울기
    BODY_TREMOR = "body_tremor"  # 몸 떨림
    PUNCH_GESTURE = "punch_gesture"  # 주먹 이스터에그


@dataclass(frozen=True)
class AlertMessage:
    """알림 메시지 Value Object - 다른 모듈에서 재사용 가능"""

    alert_type: AlertType
    message: str
    severity: str = "warning"  # warning | info | critical

    @staticmethod
    def body_tilt() -> "AlertMessage":
        return AlertMessage(AlertType.BODY_TILT, "몸을 바로 세워주세요", "warning")

    @staticmethod
    def head_tilt() -> "AlertMessage":
        return AlertMessage(AlertType.HEAD_TILT, "머리를 바로 세워주세요", "warning")

    @staticmethod
    def body_tremor() -> "AlertMessage":
        return AlertMessage(AlertType.BODY_TREMOR, "몸의 떨림이 심합니다", "warning")

    @staticmethod
    def punch_gesture() -> "AlertMessage":
        return AlertMessage(
            AlertType.PUNCH_GESTURE, "면접관을 가격해서는 안됩니다!", "critical"
        )


@dataclass
class PoseMetrics:
    """상체 포즈 지표 - Domain Entity"""

    body_tilt_angle: float  # 도 단위, 0=똑바로 (화면 중심 수직선 기준)
    head_tilt_angle: float  # 도 단위, 머리 삐딱함
    neck_head_tilt_angle: float  # 도 단위, 상체 대비 머리 기울기 (목 각도)
    tremor_level: float  # 0~1, 높을수록 떨림 심함
    is_punch_gesture: bool

    body_landmarks: Optional[dict] = None  # 디버깅/확장용
    face_landmarks: Optional[dict] = None
    hand_landmarks: Optional[dict] = None


# MediaPipe Pose 랜드마크 인덱스 (공유용)
class PoseLandmarkIndex:
    """pose_landmarks 인덱스 상수"""

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


class PoseAnalyzer:
    """
    Domain Service: 포즈 분석 비즈니스 로직
    - 몸 기울기, 머리 각도, 떨림, 주먹 감지
    - 팀원이 임계값만 수정하면 됨
    """

    # 협업 시 조정 가능한 임계값 (config로 분리해도 됨)
    BODY_TILT_THRESHOLD_DEG = 10.0  # 이 각도 이상이면 "몸을 바로 세워주세요"
    HEAD_TILT_THRESHOLD_DEG = 25.0  # 머리 삐딱함 (귀선 각도)
    NECK_HEAD_TILT_THRESHOLD_DEG = 20.0  # 상체 대비 머리 기울기 "머리를 바로 세워주세요"
    
    TREMOR_THRESHOLD = 0.005  # 이동 분산 임계값 (스무딩 적용 후)
    TREMOR_WINDOW_SIZE = 20  # 분산 계산에 사용할 스무딩된 위치 개수
    TREMOR_SMOOTH_ALPHA = 0.8  # EMA 스무딩 강도 (낮을수록 jitter 억제, 0.3~0.5 권장)
    TREMOR_ALERT_THRESHOLD = 0.85  # 이 수치 이상일 때만 알림
    TREMOR_WARMUP_FRAMES = 90  # 시작 후 N프레임(~3초)은 떨림 알림 무시
    
    PUNCH_ARM_EXTEND_RATIO = 0.85  # 팔 펴진 비율
    PUNCH_FIST_CLOSED_THRESHOLD = 0.06  # 주먹 오므림 거리 (손가락끝~밑마디 평균)

    def __init__(self, learned_punch_path: Optional[Path] = None):
        self._position_history: deque = deque(maxlen=self.TREMOR_WINDOW_SIZE)
        self._smoothed_pos: Optional[tuple[float, float]] = None  # EMA 스무딩용
        self._learned_punch_samples: list = []
        self._frame_count = 0  # 떨림 워밍업용
        if learned_punch_path and learned_punch_path.exists() and load_samples:
            self._learned_punch_samples = load_samples(learned_punch_path)

    def analyze(
        self,
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
    ) -> PoseMetrics:
        """Holistic 결과로부터 PoseMetrics 도출"""
        if pose_landmarks is None:
            return PoseMetrics(
                body_tilt_angle=0.0,
                head_tilt_angle=0.0,
                neck_head_tilt_angle=0.0,
                tremor_level=0.0,
                is_punch_gesture=False,
            )

        lm = pose_landmarks.landmark
        idx = PoseLandmarkIndex

        self._frame_count += 1
        body_tilt = self._compute_body_tilt(lm, idx)
        head_tilt = self._compute_head_tilt(lm, idx)
        neck_head_tilt = self._compute_neck_head_tilt(lm, idx)
        tremor = self._compute_tremor(lm, idx)
        punch = self._detect_punch_gesture(pose_landmarks, lm, idx, left_hand_landmarks, right_hand_landmarks)

        return PoseMetrics(
            body_tilt_angle=body_tilt,
            head_tilt_angle=head_tilt,
            neck_head_tilt_angle=neck_head_tilt,
            tremor_level=tremor,
            is_punch_gesture=punch,
        )

    def _get_xy(self, landmark) -> tuple[float, float]:
        return (landmark.x, landmark.y)

    def _compute_body_tilt(self, lm, idx) -> float:
        """
        몸 기울기: 화면 중심 수직선(x=0.5)을 기준으로 상체가 틀어진 정도.
        골반 제외. 몸 중심축(코~어깨중심)이 화면 수직선과 이루는 각도.
        """
        SCREEN_CENTER_X = 0.5
        nose = lm[idx.NOSE]
        ls = lm[idx.LEFT_SHOULDER]
        rs = lm[idx.RIGHT_SHOULDER]
        mid_shoulder_x = (ls.x + rs.x) / 2
        mid_shoulder_y = (ls.y + rs.y) / 2
        # 몸 중심축: 어깨중심 → 코 방향
        dx = nose.x - mid_shoulder_x
        dy = nose.y - mid_shoulder_y
        # 화면 수직선 방향 (아래→위) = (0, -1), 몸축이 수직과 일치하면 dx≈0
        
        # 틀어진 각도 = atan2(|dx|, |dy|)
        angle_rad = np.arctan2(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 0
        return float(np.degrees(angle_rad))

    def _compute_head_tilt(self, lm, idx) -> float:
        """머리 삐딱한 정도: 좌우 귀를 잇는 선의 수평선 대비 각도 (절대값)"""
        le = lm[idx.LEFT_EAR]
        re = lm[idx.RIGHT_EAR]
        dx = re.x - le.x
        dy = re.y - le.y
        angle_rad = np.arctan2(abs(dy), abs(dx)) if (dx != 0 or dy != 0) else 0
        return float(np.degrees(angle_rad))

    def _compute_neck_head_tilt(self, lm, idx) -> float:
        """
        목 각도: 상체(어깨선) 대비 머리(귀선)가 기울어진 정도.
        상체는 똑바로인데 머리만 삐딱한 경우를 감지.
        """
        ls = lm[idx.LEFT_SHOULDER]
        rs = lm[idx.RIGHT_SHOULDER]
        le = lm[idx.LEFT_EAR]
        re = lm[idx.RIGHT_EAR]
        # 어깨선 각도
        shoulder_dx = rs.x - ls.x
        shoulder_dy = rs.y - ls.y
        
        # 귀선 각도
        ear_dx = re.x - le.x
        ear_dy = re.y - le.y
        angle_shoulder = np.arctan2(shoulder_dy, shoulder_dx) if (shoulder_dx != 0 or shoulder_dy != 0) else 0
        angle_ear = np.arctan2(ear_dy, ear_dx) if (ear_dx != 0 or ear_dy != 0) else 0
        diff = abs(np.degrees(angle_ear - angle_shoulder))
        return float(min(diff, 180 - diff))  # 0~90도 범위

    def _compute_tremor(self, lm, idx) -> float:
        """
        [개선된 로직] 몸 떨림 감지
        1. EMA 스무딩을 강하게 적용하여 웹캠 노이즈 제거
        2. '어깨 너비'로 정규화하여 카메라 거리와 상관없이 일정하게 측정
        """
        ls = lm[idx.LEFT_SHOULDER]
        rs = lm[idx.RIGHT_SHOULDER]
        
        # 현재 어깨 중심
        cx = (ls.x + rs.x) / 2
        cy = (ls.y + rs.y) / 2

        # [추가] 거리 보정을 위한 스케일 (어깨 너비)
        # 어깨 너비가 넓으면(가까우면) 움직임도 크게 잡히므로 나눠줘야 함
        shoulder_width = np.sqrt((rs.x - ls.x)**2 + (rs.y - ls.y)**2)
        if shoulder_width == 0: shoulder_width = 1.0 # 0 나누기 방지

        # EMA 스무딩 (노이즈 억제)
        # alpha가 클수록(0.8) 이전 값 유지를 많이 하여 부드러워짐
        alpha = self.TREMOR_SMOOTH_ALPHA
        if self._smoothed_pos is None:
            self._smoothed_pos = (cx, cy)
        else:
            self._smoothed_pos = (
                alpha * self._smoothed_pos[0] + (1 - alpha) * cx,
                alpha * self._smoothed_pos[1] + (1 - alpha) * cy,
            )
        
        # 스무딩된 위치 저장
        self._position_history.append(self._smoothed_pos)

        if len(self._position_history) < self.TREMOR_WINDOW_SIZE:
            return 0.0

        # 분산(Variance) 계산
        arr = np.array(self._position_history)
        var_x = np.var(arr[:, 0])
        var_y = np.var(arr[:, 1])
        total_var = var_x + var_y

        # [핵심 수정] 스케일 정규화 (Scale Invariant)
        # 단순히 좌표 분산만 보면 가까이 있을 때 값이 튀므로, 어깨 너비 제곱으로 나눔
        normalized_tremor = total_var / (shoulder_width ** 2)

        # 0~1 스케일링 (Log 스케일을 적용하면 더 민감도 조절이 쉽지만, 여기선 선형 유지)
        # 기준치(TREMOR_THRESHOLD)보다 분산이 크면 1.0에 가까워짐
        level = min(1.0, normalized_tremor / self.TREMOR_THRESHOLD)
        
        return float(level)

    def _detect_punch_gesture(
        self,
        pose_landmarks,
        lm,
        idx,
        left_hand,
        right_hand,
    ) -> bool:
        """
        이스터에그: 화면을 향해 주먹을 날리는 자세
        - 규칙 기반: 팔 펴짐 + 주먹 + 카메라 방향
        - 학습 기반: pose_learn으로 저장한 샘플과 유사하면 감지
        """
        # 학습된 샘플이 있으면 먼저 확인
        if self._learned_punch_samples and is_punch_from_learned:
            if is_punch_from_learned(pose_landmarks, left_hand, right_hand, self._learned_punch_samples, threshold=0.65):
                return True
        # 규칙 기반
        for side, wrist_idx, elbow_idx, hand_lm in [
            ("left", idx.LEFT_WRIST, idx.LEFT_ELBOW, left_hand),
            ("right", idx.RIGHT_WRIST, idx.RIGHT_ELBOW, right_hand),
        ]:
            if hand_lm is None:
                continue

            wrist = lm[wrist_idx]
            elbow = lm[elbow_idx]
            shoulder = lm[idx.LEFT_SHOULDER if side == "left" else idx.RIGHT_SHOULDER]

            # 1) 팔이 펴져 있는지: wrist-elbow-shoulder 각도
            arm_extended = self._is_arm_extended(wrist, elbow, shoulder)
            if not arm_extended:
                continue

            # 2) 주먹인지: 손 랜드마크에서 손가락이 오므려져 있는지
            if not self._is_fist_closed(hand_lm):
                continue

            # 3) 화면 방향(카메라 쪽): wrist.z가 shoulder보다 작으면 앞에 있음
            if wrist.z < shoulder.z:
                return True

        return False

    def _is_arm_extended(self, wrist, elbow, shoulder) -> bool:
        """팔이 거의 직선으로 펴져 있는지"""
        v1 = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        return angle > 150  # 거의 180도 = 팔 펴짐

    def _is_fist_closed(self, hand_landmarks) -> bool:
        """주먹 쥔 상태: 손가락 끝이 손바닥 쪽으로 접혀있음"""
        if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
            return False
        # MediaPipe Hand: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        # 2,5,9,13,17 = 각 손가락의 첫 번째 마디
        wrist = hand_landmarks.landmark[0]
        fingertips = [4, 8, 12, 16, 20]
        mcp = [2, 5, 9, 13, 17]  # 손가락 밑마디
        total_dist = 0.0
        for tip_i, mcp_i in zip(fingertips, mcp):
            t = hand_landmarks.landmark[tip_i]
            m = hand_landmarks.landmark[mcp_i]
            d = np.sqrt((t.x - m.x) ** 2 + (t.y - m.y) ** 2)
            total_dist += d
        avg = total_dist / 5
        return avg < self.PUNCH_FIST_CLOSED_THRESHOLD


# =============================================================================
# APPLICATION LAYER - 유스케이스
# =============================================================================


class AlertChecker:
    """Application Service: 분석 결과 → 알림 여부 판단"""

    def __init__(self, analyzer: PoseAnalyzer):
        self._analyzer = analyzer

    def check_alerts(self, metrics: PoseMetrics) -> list[AlertMessage]:
        alerts = []
        if metrics.body_tilt_angle >= self._analyzer.BODY_TILT_THRESHOLD_DEG:
            alerts.append(AlertMessage.body_tilt())
        # 머리/목 기울기: 귀선 각도 또는 상체 대비 머리 기울기
        if (
            metrics.head_tilt_angle >= self._analyzer.HEAD_TILT_THRESHOLD_DEG
            or metrics.neck_head_tilt_angle >= self._analyzer.NECK_HEAD_TILT_THRESHOLD_DEG
        ):
            alerts.append(AlertMessage.head_tilt())
        # 떨림: 워밍업 후 + 임계값 이상일 때만 (정말 긴장한 수준)
        if (
            metrics.tremor_level >= self._analyzer.TREMOR_ALERT_THRESHOLD
            and self._analyzer._frame_count > self._analyzer.TREMOR_WARMUP_FRAMES
        ):
            alerts.append(AlertMessage.body_tremor())
        if metrics.is_punch_gesture:
            alerts.append(AlertMessage.punch_gesture())
        return alerts


# =============================================================================
# INFRASTRUCTURE LAYER - 외부 의존성 (MediaPipe)
# =============================================================================


class HolisticBackbone(ABC):
    """백본 추상 인터페이스 - 팀원이 다른 백본으로 교체 가능"""

    @abstractmethod
    def process(self, image_rgb) -> "HolisticResult":
        pass


@dataclass
class HolisticResult:
    """Holistic 추론 결과 - 도메인/인프라 경계 데이터"""

    pose_landmarks: Optional[object]
    face_landmarks: Optional[object]
    left_hand_landmarks: Optional[object]
    right_hand_landmarks: Optional[object]


class MediaPipeHolisticBackbone(HolisticBackbone):
    """MediaPipe Holistic 백본 구현체 (media_pipe.py 기반)"""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, image_rgb) -> HolisticResult:
        results = self._holistic.process(image_rgb)
        return HolisticResult(
            pose_landmarks=results.pose_landmarks,
            face_landmarks=results.face_landmarks,
            left_hand_landmarks=results.left_hand_landmarks,
            right_hand_landmarks=results.right_hand_landmarks,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._holistic.close()


# =============================================================================
# INTERFACE LAYER - 알림 표시 및 메인 루프
# =============================================================================


def _put_text_korean(frame: np.ndarray, text: str, x: int, y: int, color=(255, 255, 255), font_size: int = 24, center_x: Optional[int] = None) -> np.ndarray:
    """
    한글 텍스트를 OpenCV 프레임에 그립니다.
    cv2.putText는 한글을 지원하지 않으므로 PIL로 렌더링 후 합성합니다.
    center_x가 주어지면 해당 x를 기준으로 텍스트를 가운데 정렬합니다.
    """
    if not _HAS_PIL or not text:
        cv2.putText(frame, text or "?", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return frame

    # macOS/Windows/Linux에서 한글 폰트 탐색
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",  # Windows
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
    ]
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw_x = (center_x - tw // 2) if center_x is not None else x
    draw_y = y
    draw.text((draw_x, draw_y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


class AlertPresenter(ABC):
    """알림 표시 추상 인터페이스 - GUI/오버레이 등으로 확장 가능"""

    @abstractmethod
    def show(self, message: AlertMessage, frame: np.ndarray) -> np.ndarray:
        pass


class OverlayAlertPresenter(AlertPresenter):
    """OpenCV 오버레이로 알림창 표시 - 알림당 3~5초 유지"""

    DISPLAY_DURATION_FRAMES = 90  # 3초 @ 30fps (3~5초 범위)

    def __init__(self):
        self._frame_count = 0
        # (AlertMessage, 만료 프레임) - 같은 타입은 재트리거 시 만료 연장
        self._active_alerts: dict[AlertType, tuple[AlertMessage, int]] = {}

    def update_and_show(self, alerts: list[AlertMessage], frame: np.ndarray) -> np.ndarray:
        """
        새로 발생한 알림을 추가하고, 모든 활성 알림을 표시.
        각 알림은 마지막 트리거 시점부터 DISPLAY_DURATION_FRAMES 동안 표시됨.
        """
        self._frame_count += 1
        now = self._frame_count
        expiry = now + self.DISPLAY_DURATION_FRAMES

        for msg in alerts:
            self._active_alerts[msg.alert_type] = (msg, expiry)

        # 만료된 알림 제거
        self._active_alerts = {
            k: v for k, v in self._active_alerts.items()
            if v[1] > now
        }

        # 활성 알림들을 위에서부터 순서대로 표시
        h, w = frame.shape[:2]
        pad = 20
        box_h = 50
        line_height = 55

        for i, (msg, _) in enumerate(self._active_alerts.values()):
            y1 = pad + i * line_height
            y2 = y1 + box_h
            x1, x2 = pad, w - pad

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            alpha = 0.85
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # 한글 지원: PIL로 렌더링
            center_x = w // 2
            ty = y1 + 12
            frame = _put_text_korean(frame, msg.message, 0, ty, color=(255, 255, 255), font_size=22, center_x=center_x)
        return frame

    def show(self, message: AlertMessage, frame: np.ndarray) -> np.ndarray:
        """기존 인터페이스 호환 - 단일 알림을 active에 추가 후 draw"""
        return self.update_and_show([message], frame)


def run_pose_feedback_pipeline():
    """
    메인 진입점: Holistic 백본 → 포즈 분석 → 알림 판단 → 오버레이
    팀원이 face/gesture 파이프라인을 이 구조에 맞춰 합치면 됨
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 찾을 수 없습니다.")
        return

    analyzer = PoseAnalyzer(learned_punch_path=LEARNED_PUNCH_PATH)
    alert_checker = AlertChecker(analyzer)
    alert_presenter = OverlayAlertPresenter()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 랜드마크 그리기 (선택)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # 포즈 분석 및 알림
            metrics = analyzer.analyze(
                results.pose_landmarks,
                results.left_hand_landmarks,
                results.right_hand_landmarks,
            )
            alerts = alert_checker.check_alerts(metrics)
            image = alert_presenter.update_and_show(alerts, image)

            # 상태 디버그 (선택 사항)
            cv2.putText(image, f"Tilt:{metrics.body_tilt_angle:.1f} Tremor:{metrics.tremor_level:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Interview Feedback - Pose (Test2_MP)", image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# 팀 협업용: 외부에서 백본만 주입하여 사용
# =============================================================================


def create_pose_feedback_service(backbone: HolisticBackbone) -> tuple[PoseAnalyzer, AlertChecker]:
    """
    다른 팀원이 자체 백본/파이프라인과 연동할 때 사용.
    예: feedback_service = create_pose_feedback_service(my_backbone)
    """
    analyzer = PoseAnalyzer()
    checker = AlertChecker(analyzer)
    return analyzer, checker


if __name__ == "__main__":
    run_pose_feedback_pipeline()
