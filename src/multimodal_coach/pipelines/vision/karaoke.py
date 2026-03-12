"""
유명인사 연설 따라하기 (Speech Karaoke) - 핵심 로직 모듈

MediaPipe Holistic을 활용하여 원본 영상과 사용자 웹캠을 비교,
자세(Pose), 손동작(Hands), 얼굴 각도(Face)의 유사도를 실시간으로 분석합니다.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# =============================================================================
# 데이터 구조 및 상수
# =============================================================================


@dataclass
class FrameLandmarkData:
    """
    단일 프레임의 정규화된 랜드마크 및 핵심 벡터 데이터
    - 어깨 너비 기준 정규화로 카메라 거리/체형 차이 보정
    """

    frame_idx: int
    timestamp_ms: float
    # 정규화된 포즈 벡터들 (어깨 너비 = 1.0 기준)
    shoulder_elbow_wrist_vectors: dict  # {'left': [...], 'right': [...]}
    head_tilt_angles: dict  # {'roll': float, 'pitch': float, 'yaw': float}
    hand_open_ratios: dict  # {'left': float, 'right': float}  # 0=주먹, 1=손펴짐
    shoulder_center: tuple[float, float]
    shoulder_width: float
    # 트레모 측정용: 원본 좌표 (정규화 전)
    raw_shoulder_positions: tuple
    raw_wrist_positions: tuple


# MediaPipe Pose 랜드마크 인덱스 (33개)
class PoseLandmarkIndex:
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


# MediaPipe Hand 랜드마크 인덱스 (21개): 0=손목, 4=엄지끝, 8=검지끝, 12=중지끝, 16=약지끝, 20=소지끝
# 2,5,9,13,17 = 각 손가락 첫째 마디(MCP)
HAND_FINGERTIPS = [4, 8, 12, 16, 20]
HAND_MCP = [2, 5, 9, 13, 17]


# =============================================================================
# Speech Karaoke Trainer - 핵심 클래스
# =============================================================================


class SpeechKaraokeTrainer:
    """
    유명인사 연설 따라하기 트레이너
    - Step 1: 원본 영상에서 Teacher Data 추출 → JSON 저장
    - Step 2: 실시간 웹캠과 Teacher Data 비교 → 점수 계산
    """

    def __init__(
        self,
        tremor_window_size: int = 30,
        tremor_smooth_alpha: float = 0.7,
        tremor_noise_threshold: float = 0.003,
    ):
        """
        Args:
            tremor_window_size: 떨림 분산 계산에 사용할 프레임 수 (기본 30프레임)
            tremor_smooth_alpha: EMA 스무딩 알파 (낮을수록 노이즈 억제 강함)
            tremor_noise_threshold: 이 값 미만이면 노이즈로 간주 (진짜 떨림과 구분)
        """
        self.tremor_window_size = tremor_window_size
        self.tremor_smooth_alpha = tremor_smooth_alpha
        self.tremor_noise_threshold = tremor_noise_threshold
        # MediaPipe Holistic 초기화
        self._holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._holistic.close()

    # -------------------------------------------------------------------------
    # Step 1: Teacher Data 추출
    # -------------------------------------------------------------------------

    def extract_reference_data(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        output_format: str = "json",
    ) -> list[dict]:
        """
        원본 영상(mp4)에서 프레임별 정규화된 랜드마크 및 핵심 벡터를 추출하여 저장합니다.

        Args:
            video_path: 입력 영상 경로 (mp4)
            output_path: 저장 경로 (미지정 시 video_path와 동일한 이름으로 .json/.csv 저장)
            output_format: 'json' 또는 'csv'

        Returns:
            프레임별 추출 데이터 리스트 (dict 리스트)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval_ms = 1000.0 / fps

        extracted_data: list[dict] = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # BGR → RGB 변환 (MediaPipe 입력 형식)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._holistic.process(frame_rgb)

            timestamp_ms = frame_idx * frame_interval_ms

            # 포즈/얼굴/손 랜드마크로부터 정규화된 데이터 추출
            frame_data = self._extract_single_frame_data(
                results, frame_idx, timestamp_ms
            )

            if frame_data is not None:
                # dict로 직렬화 가능하게 변환
                extracted_data.append(self._frame_data_to_dict(frame_data))

            frame_idx += 1

        cap.release()

        # 저장 경로 설정
        if output_path is None:
            output_path = video_path.with_suffix(f".{output_format}")
        else:
            output_path = Path(output_path)

        if output_format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"fps": fps, "frames": extracted_data},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        elif output_format.lower() == "csv":
            self._save_to_csv(extracted_data, output_path, fps)
        else:
            raise ValueError("output_format은 'json' 또는 'csv' 여야 합니다.")

        return extracted_data

    def _extract_single_frame_data(
        self, results, frame_idx: int, timestamp_ms: float
    ) -> Optional[FrameLandmarkData]:
        """
        Holistic 결과에서 단일 프레임의 정규화된 데이터를 추출합니다.
        """
        pose = results.pose_landmarks
        if pose is None:
            return None

        lm = pose.landmark
        idx = PoseLandmarkIndex

        # 어깨 중심 및 너비 (정규화 기준)
        ls = lm[idx.LEFT_SHOULDER]
        rs = lm[idx.RIGHT_SHOULDER]
        shoulder_center = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        shoulder_width = np.sqrt(
            (rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2
        ) or 1.0

        # Body Pose: 어깨-팔꿈치-손목 벡터 (정규화)
        shoulder_elbow_wrist_vectors = {}
        for side, s_i, e_i, w_i in [
            ("left", idx.LEFT_SHOULDER, idx.LEFT_ELBOW, idx.LEFT_WRIST),
            ("right", idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_WRIST),
        ]:
            v1 = self._normalize_vector(
                lm[e_i], lm[s_i], shoulder_width
            )  # 어깨→팔꿈치
            v2 = self._normalize_vector(
                lm[w_i], lm[e_i], shoulder_width
            )  # 팔꿈치→손목
            shoulder_elbow_wrist_vectors[side] = [
                v1.tolist(),
                v2.tolist(),
            ]

        # Head Tilt: 콧대와 어깨선이 이루는 각도 (Roll/Pitch/Yaw)
        head_tilt_angles = self._compute_head_tilt_angles(lm, idx, shoulder_width)

        # Hand Status: 손바닥 개폐 비율 (0=주먹, 1=손 펴짐)
        hand_open_ratios = {}
        for hand_name, hand_lm in [
            ("left", results.left_hand_landmarks),
            ("right", results.right_hand_landmarks),
        ]:
            hand_open_ratios[hand_name] = self._compute_hand_open_ratio(hand_lm)

        # 트레모 측정용 원본 좌표
        raw_shoulder = ((ls.x, ls.y), (rs.x, rs.y))
        lw = lm[idx.LEFT_WRIST]
        rw = lm[idx.RIGHT_WRIST]
        raw_wrist = ((lw.x, lw.y), (rw.x, rw.y))

        return FrameLandmarkData(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            shoulder_elbow_wrist_vectors=shoulder_elbow_wrist_vectors,
            head_tilt_angles=head_tilt_angles,
            hand_open_ratios=hand_open_ratios,
            shoulder_center=shoulder_center,
            shoulder_width=shoulder_width,
            raw_shoulder_positions=raw_shoulder,
            raw_wrist_positions=raw_wrist,
        )

    def _normalize_vector(self, p_end, p_start, scale: float) -> np.ndarray:
        """두 점으로 벡터 생성 후 scale로 나누어 정규화 (체형 보정)"""
        v = np.array(
            [
                p_end.x - p_start.x,
                p_end.y - p_start.y,
                getattr(p_end, "z", 0) - getattr(p_start, "z", 0),
            ]
        )
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.zeros(3)
        return v / (scale if scale > 0 else 1.0)

    def _compute_head_tilt_angles(
        self, lm, idx, shoulder_width: float
    ) -> dict[str, float]:
        """
        콧대와 어깨선이 이루는 각도 (Roll/Pitch/Yaw 근사)
        - Roll: 귀선의 수평 대비 기울기 (머리 좌우 기울기)
        - Pitch: 코-어깨중심 벡터의 수직 대비 각도 (끄덕임)
        - Yaw: 코가 화면 중심에서 벗어난 정도 (고개 돌림)
        """
        nose = lm[idx.NOSE]
        le = lm[idx.LEFT_EAR]
        re = lm[idx.RIGHT_EAR]
        ls = lm[idx.LEFT_SHOULDER]
        rs = lm[idx.RIGHT_SHOULDER]
        mid_shoulder_x = (ls.x + rs.x) / 2
        mid_shoulder_y = (ls.y + rs.y) / 2

        # Roll: 귀선 각도
        ear_dx = re.x - le.x
        ear_dy = re.y - le.y
        roll = np.degrees(
            np.arctan2(ear_dy, ear_dx)
            if (abs(ear_dx) > 1e-6 or abs(ear_dy) > 1e-6)
            else 0
        )

        # Pitch: 코→어깨중심 벡터가 아래 방향(0,1)과 이루는 각도
        dx = nose.x - mid_shoulder_x
        dy = nose.y - mid_shoulder_y
        pitch = (
            np.degrees(np.arctan2(dx, dy))
            if (abs(dx) > 1e-6 or abs(dy) > 1e-6)
            else 0
        )

        # Yaw: 코의 x좌표가 어깨중심에서 얼마나 벗어났는지 (정규화)
        yaw = (nose.x - mid_shoulder_x) / (shoulder_width or 1.0) * 90.0

        return {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)}

    def _compute_hand_open_ratio(self, hand_landmarks) -> float:
        """
        손바닥 개폐 비율 (0=주먹, 1=손 완전히 펴짐)
        손가락 끝과 밑마디 거리의 평균을 0~1 사이로 정규화
        """
        if hand_landmarks is None or len(hand_landmarks.landmark) < 21:
            return 0.5  # 감지 안 됨 시 중간값

        total_dist = 0.0
        for tip_i, mcp_i in zip(HAND_FINGERTIPS, HAND_MCP):
            t = hand_landmarks.landmark[tip_i]
            m = hand_landmarks.landmark[mcp_i]
            d = np.sqrt((t.x - m.x) ** 2 + (t.y - m.y) ** 2)
            total_dist += d
        avg_dist = total_dist / 5
        # 경험적 범위: 0.02~0.12 정도 → 0~1로 스케일
        ratio = np.clip((avg_dist - 0.02) / 0.10, 0.0, 1.0)
        return float(ratio)

    def _frame_data_to_dict(self, fd: FrameLandmarkData) -> dict:
        """FrameLandmarkData를 JSON 직렬화 가능한 dict로 변환"""
        return {
            "frame_idx": fd.frame_idx,
            "timestamp_ms": fd.timestamp_ms,
            "shoulder_elbow_wrist_vectors": fd.shoulder_elbow_wrist_vectors,
            "head_tilt_angles": fd.head_tilt_angles,
            "hand_open_ratios": fd.hand_open_ratios,
            "shoulder_center": list(fd.shoulder_center),
            "shoulder_width": fd.shoulder_width,
        }

    def _save_to_csv(
        self, extracted_data: list[dict], output_path: Path, fps: float
    ) -> None:
        """추출 데이터를 CSV 형식으로 저장 (간소화된 주요 컬럼)"""
        import csv

        if not extracted_data:
            return

        # 프레임 인덱스, 타임스탬프, 주요 각도/비율만 flatten
        rows = []
        for d in extracted_data:
            row = {
                "frame_idx": d["frame_idx"],
                "timestamp_ms": d["timestamp_ms"],
                "head_roll": d["head_tilt_angles"]["roll"],
                "head_pitch": d["head_tilt_angles"]["pitch"],
                "head_yaw": d["head_tilt_angles"]["yaw"],
                "hand_left": d["hand_open_ratios"]["left"],
                "hand_right": d["hand_open_ratios"]["right"],
            }
            rows.append(row)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # -------------------------------------------------------------------------
    # Step 2: 실시간 유사도 계산
    # -------------------------------------------------------------------------

    def calculate_pose_similarity(
        self,
        user_landmarks: dict,
        ref_landmarks: dict,
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """
        사용자 랜드마크와 참조(Teacher) 랜드마크를 코사인 유사도로 비교하여
        일치율(%)을 반환합니다. 어깨 너비 기준 정규화는 호출 전 수행된 상태를 가정합니다.

        Args:
            user_landmarks: _frame_data_to_dict 형식의 사용자 프레임 데이터
            ref_landmarks: 참조(원본) 영상의 동일 형식 데이터
            weights: {'body_pose': 0.4, 'head_tilt': 0.3, 'hand_status': 0.3} 등
                     미지정 시 동일 비율

        Returns:
            0.0 ~ 1.0 (100% = 1.0) 유사도
        """
        if weights is None:
            weights = {"body_pose": 0.4, "head_tilt": 0.3, "hand_status": 0.3}

        scores = []

        # 1) Body Pose: 어깨-팔꿈치-손목 벡터들의 코사인 유사도
        body_sim = self._cosine_similarity_vectors(
            user_landmarks.get("shoulder_elbow_wrist_vectors", {}),
            ref_landmarks.get("shoulder_elbow_wrist_vectors", {}),
        )
        scores.append(
            (weights.get("body_pose", 1.0 / 3), body_sim)
        )

        # 2) Head Tilt: Roll/Pitch/Yaw 각도 유사도 (각도 차이를 유사도로 변환)
        head_sim = self._angle_similarity(
            user_landmarks.get("head_tilt_angles", {}),
            ref_landmarks.get("head_tilt_angles", {}),
        )
        scores.append(
            (weights.get("head_tilt", 1.0 / 3), head_sim)
        )

        # 3) Hand Status: 손 개폐 비율 유사도 (1 - 절대차)
        hand_sim = self._hand_ratio_similarity(
            user_landmarks.get("hand_open_ratios", {}),
            ref_landmarks.get("hand_open_ratios", {}),
        )
        scores.append(
            (weights.get("hand_status", 1.0 / 3), hand_sim)
        )

        # 가중 평균
        total_w = sum(w for w, _ in scores)
        if total_w < 1e-8:
            return 0.0
        return sum(w * s for w, s in scores) / total_w

    def _cosine_similarity_vectors(
        self, user_vectors: dict, ref_vectors: dict
    ) -> float:
        """양쪽 벡터들(어깨-팔꿈치-손목)의 코사인 유사도 평균"""
        sims = []
        for side in ["left", "right"]:
            u_vecs = user_vectors.get(side, [])
            r_vecs = ref_vectors.get(side, [])
            if not u_vecs or not r_vecs:
                continue
            for uv, rv in zip(u_vecs, r_vecs):
                uv = np.array(uv)
                rv = np.array(rv)
                if np.linalg.norm(uv) < 1e-8 or np.linalg.norm(rv) < 1e-8:
                    sims.append(1.0)
                    continue
                cos_sim = np.dot(uv, rv) / (
                    np.linalg.norm(uv) * np.linalg.norm(rv)
                )
                cos_sim = np.clip(cos_sim, -1, 1)
                # -1~1 → 0~1 로 변환
                sims.append((cos_sim + 1) / 2)
        return float(np.mean(sims)) if sims else 0.5

    def _angle_similarity(
        self, user_angles: dict, ref_angles: dict
    ) -> float:
        """각도 차이를 0~1 유사도로 변환 (차이가 0도면 1, 90도면 0 근처)"""
        sims = []
        for key in ["roll", "pitch", "yaw"]:
            u = user_angles.get(key, 0)
            r = ref_angles.get(key, 0)
            diff = abs(u - r)
            diff = min(diff, 360 - diff)
            # 90도 차이 = 0, 0도 차이 = 1
            s = max(0, 1 - diff / 90.0)
            sims.append(s)
        return float(np.mean(sims)) if sims else 0.5

    def _hand_ratio_similarity(
        self, user_ratios: dict, ref_ratios: dict
    ) -> float:
        """손 개폐 비율 유사도 (1 - 평균 절대 차이)"""
        sims = []
        for side in ["left", "right"]:
            u = user_ratios.get(side, 0.5)
            r = ref_ratios.get(side, 0.5)
            sims.append(1 - min(1, abs(u - r)))
        return float(np.mean(sims)) if sims else 0.5

    # -------------------------------------------------------------------------
    # Tremor (떨림) 감지
    # -------------------------------------------------------------------------

    def detect_tremor(
        self,
        landmark_history: list[tuple] | deque,
        use_smoothing: bool = True,
    ) -> float:
        """
        특정 윈도우 내에서 손 또는 어깨 랜드마크 위치 변화의 분산(Variance)을 측정하여
        떨림 수준을 반환합니다. 노이즈와 구분하기 위해 이동 평균(EMA) 스무딩을 적용합니다.

        Args:
            landmark_history: (x, y) 또는 (x, y, z) 튜플의 리스트/데크
                             예: [(0.5, 0.3), (0.51, 0.31), ...]
            use_smoothing: True면 EMA 스무딩 적용 후 분산 계산 (노이즈 억제)

        Returns:
            0.0 ~ 1.0 (높을수록 떨림 심함). tremor_noise_threshold 미만이면 0에 가깝게 반환
        """
        if len(landmark_history) < self.tremor_window_size:
            return 0.0

        # 최근 N개만 사용
        recent = list(landmark_history)[-self.tremor_window_size :]
        arr = np.array(recent)

        if use_smoothing:
            # EMA 스무딩 적용 (노이즈 제거, 진짜 떨림만 강조)
            smoothed = []
            alpha = self.tremor_smooth_alpha
            s = np.array(arr[0], dtype=float)
            for i in range(len(arr)):
                s = alpha * s + (1 - alpha) * arr[i]
                smoothed.append(s.copy())
            arr = np.array(smoothed)

        # 분산 계산 (x, y)
        var_x = np.var(arr[:, 0])
        var_y = np.var(arr[:, 1])
        total_var = var_x + var_y

        # 노이즈 임계값 미만이면 0으로 간주
        if total_var < self.tremor_noise_threshold:
            return 0.0

        # 0~1 스케일 (임계값의 5배 정도를 1.0으로)
        level = min(
            1.0,
            (total_var - self.tremor_noise_threshold)
            / (self.tremor_noise_threshold * 5),
        )
        return float(level)

    # -------------------------------------------------------------------------
    # 실시간 스트림용 헬퍼
    # -------------------------------------------------------------------------

    def normalize_user_frame(
        self, holistic_results, shoulder_width_ref: float = 1.0
    ) -> Optional[dict]:
        """
        웹캠에서 얻은 Holistic 결과를 참조 영상과 같은 방식으로 정규화하여
        calculate_pose_similarity에 넣을 수 있는 dict를 반환합니다.
        shoulder_width_ref: 참조 영상의 평균 어깨 너비 (선택, 기본 1.0)
        """
        frame_data = self._extract_single_frame_data(
            holistic_results, 0, 0.0
        )
        if frame_data is None:
            return None
        return self._frame_data_to_dict(frame_data)

    def get_ref_frame_by_timestamp(
        self, ref_data: list[dict], timestamp_ms: float
    ) -> Optional[dict]:
        """타임스탬프에 가장 가까운 참조 프레임을 반환 (동기화용)"""
        if not ref_data:
            return None
        idx = min(
            range(len(ref_data)),
            key=lambda i: abs(ref_data[i]["timestamp_ms"] - timestamp_ms),
        )
        return ref_data[idx]


# =============================================================================
# 사용 예시 (실행 시)
# =============================================================================


def _load_subtitles(subs_path: str | Path) -> list[dict]:
    """자막 JSON 로드 (start_sec, end_sec, text). 없으면 빈 리스트."""
    p = Path(subs_path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _get_subtitle_segments_at_time(
    subs: list[dict], current_sec: float
) -> tuple[list[dict], Optional[dict], list[dict]]:
    """
    현재 시간 기준 과거/현재/다음 자막 구간 반환.
    Returns: (past_segments, current_segment, next_segments)
    """
    past, current, following = [], None, []
    for seg in subs:
        start = seg.get("start_sec", 0)
        end = seg.get("end_sec", float("inf"))
        if current_sec >= end:
            past.append(seg)
        elif start <= current_sec < end:
            current = seg
        else:
            following.append(seg)
    return past, current, following


def _wrap_text(text: str, chars_per_line: int = 42) -> list[str]:
    """텍스트를 지정 길이로 줄바꿈."""
    lines = []
    words = text.split()
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= chars_per_line:
            current = (current + " " + w).strip() if current else w
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _draw_subtitle_karaoke(
    img: np.ndarray,
    subs: list[dict],
    current_sec: float,
    y_bottom: int,
    max_width: int,
) -> None:
    """
    노래방 스타일 자막: 과거(회색), 현재(강조+진행바), 다음(흐림).
    """
    past, current, next_segs = _get_subtitle_segments_at_time(subs, current_sec)
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    pad = 6
    progress_bar_h = 6

    all_lines = []
    for seg in past[-1:]:  # 과거: 마지막 1개만 표시 (회색)
        all_lines.append(("past", _wrap_text(seg.get("text", ""))))
    if current:  # 현재: 강조
        all_lines.append(("current", _wrap_text(current.get("text", ""))))
    for seg in next_segs[:1]:  # 다음: 1개만 (흐림)
        all_lines.append(("next", _wrap_text(seg.get("text", ""))))

    if not all_lines:
        return

    total_h = sum(len(lines) * line_height + pad for _, lines in all_lines) + progress_bar_h
    box_y1 = max(0, y_bottom - total_h)
    box_y2 = y_bottom

    overlay = img.copy()
    cv2.rectangle(overlay, (0, box_y1), (max_width, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    y_cursor = box_y1 + pad
    current_box_y1, current_box_y2 = None, None

    for seg_type, lines in all_lines:
        for line in lines:
            if seg_type == "past":
                color = (120, 120, 120)
                scale = 0.45
            elif seg_type == "current":
                color = (255, 255, 100)
                scale = 0.6
                if current_box_y1 is None:
                    current_box_y1 = y_cursor - 2
                current_box_y2 = y_cursor + line_height
            else:
                color = (180, 180, 180)
                scale = 0.5

            cv2.putText(
                img, line, (12, y_cursor + line_height - 4),
                font, scale, color, 1, cv2.LINE_AA
            )
            y_cursor += line_height + 2

        y_cursor += pad

    # 현재 구간 왼쪽 녹색 인디케이터 + 진행바
    if current and "start_sec" in current and "end_sec" in current:
        start_s = current["start_sec"]
        end_s = current["end_sec"]
        if end_s > start_s:
            progress = (current_sec - start_s) / (end_s - start_s)
            progress = max(0, min(1, progress))
            cv2.rectangle(img, (4, box_y1), (6, box_y2), (0, 200, 100), -1)
            bar_x1, bar_x2 = 10, max_width - 10
            bar_y = box_y2 - progress_bar_h - 4
            cv2.rectangle(img, (bar_x1, bar_y), (bar_x2, bar_y + progress_bar_h), (60, 60, 60), -1)
            cv2.rectangle(img, (bar_x1, bar_y), (int(bar_x1 + (bar_x2 - bar_x1) * progress), bar_y + progress_bar_h), (0, 200, 100), -1)


def _example_usage():
    """웹캠 + 참조 JSON으로 실시간 유사도 비교 (참조 영상 표시 + 가이드 + 자막)"""
    ref_video_path = "Obama's 2004 DNC keynote speech.mp4"
    ref_json_path = "Obama's 2004 DNC keynote speech.json"
    subs_path = "Obama's 2004 DNC keynote speech_subs.json"

    with open(ref_json_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    ref_frames = ref["frames"]
    fps = ref["fps"]
    subtitles = _load_subtitles(subs_path)

    cap_webcam = cv2.VideoCapture(0)
    cap_ref = cv2.VideoCapture(ref_video_path)
    if not cap_webcam.isOpened():
        print("웹캠을 찾을 수 없습니다.")
        return
    if not cap_ref.isOpened():
        print(f"참조 영상을 찾을 수 없습니다: {ref_video_path}")
        return

    print("Speech Karaoke 시작. 키: 1(0.5x), 2(1x), 3(1.25x), 4(1.5x), 5(2x), q(종료)")

    with SpeechKaraokeTrainer() as trainer:
        win_name = "Speech Karaoke - Follow Obama!"
        start_time = time.time()
        speed_multiplier = 1.0
        SPEED_OPTIONS = {ord("1"): 0.5, ord("2"): 1.0, ord("3"): 1.25, ord("4"): 1.5, ord("5"): 2.0}

        while cap_webcam.isOpened():
            success_web, image_web = cap_webcam.read()
            if not success_web:
                break

            # 경과 시간 기반 (실제 시간, 속도 배율 적용)
            elapsed_sec = time.time() - start_time
            effective_sec = elapsed_sec * speed_multiplier
            total_duration_sec = ref_frames[-1]["timestamp_ms"] / 1000.0 if ref_frames else 36.0
            effective_sec = effective_sec % total_duration_sec
            timestamp_ms = effective_sec * 1000

            ref_idx = int((timestamp_ms / 1000) * fps) % len(ref_frames)
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            success_ref, image_ref = cap_ref.read()
            if not success_ref:
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success_ref, image_ref = cap_ref.read()

            # 웹캠 거울 상 (자세 따라하기 용이)
            image_web_mirror = cv2.flip(image_web, 1)

            # 두 영상 크기 맞추기 (가로로 나란히 배치)
            h_web, w_web = image_web_mirror.shape[:2]
            if success_ref and image_ref is not None:
                image_ref = cv2.resize(
                    image_ref, (w_web, h_web), interpolation=cv2.INTER_LINEAR
                )
            else:
                image_ref = np.zeros((h_web, w_web, 3), dtype=np.uint8)
                image_ref[:] = (40, 40, 40)

            # 합치기: [참조 영상 | 웹캠(거울상)]
            combined = np.hstack([image_ref, image_web_mirror])
            h_comb, w_comb = combined.shape[:2]

            # 레이블 표시
            cv2.rectangle(combined, (0, 0), (w_web, 35), (60, 60, 60), -1)
            cv2.rectangle(combined, (w_web, 0), (w_comb, 35), (40, 80, 40), -1)
            cv2.putText(
                combined, "[REF] Obama - follow this pose",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                combined, "[YOU] Your webcam (mirror)",
                (w_web + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            # 자막: 노래방 스타일 (과거/현재/다음 + 진행바)
            if subtitles:
                _draw_subtitle_karaoke(combined, subtitles, effective_sec, h_web, w_web)

            # 유사도 계산 (원본 웹캠 사용 - 거울상은 표시용만)
            image_rgb = cv2.cvtColor(image_web, cv2.COLOR_BGR2RGB)
            results = trainer._holistic.process(image_rgb)
            user_dict = trainer.normalize_user_frame(results)
            similarity_pct = 0.0

            if user_dict and ref_frames:
                ref_frame = ref_frames[ref_idx]
                similarity_pct = trainer.calculate_pose_similarity(user_dict, ref_frame) * 100

            # 유사도 바 + 설명 (오른쪽 웹캠 영역에)
            bar_x, bar_y = w_web + 20, h_comb - 120
            bar_w, bar_h = w_web - 40, 30
            cv2.rectangle(combined, (bar_x, bar_y - 5), (bar_x + bar_w, bar_y + bar_h + 5), (80, 80, 80), -1)
            fill_w = int(bar_w * (similarity_pct / 100))
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 100), -1)
            cv2.putText(
                combined, f"Similarity: {similarity_pct:.1f}%",
                (bar_x, bar_y + bar_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2
            )
            cv2.putText(
                combined, "Left: reference | Match pose | 1-5: speed | q: quit",
                (bar_x, bar_y + bar_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
            )
            cv2.putText(
                combined, f"Speed: {speed_multiplier}x",
                (bar_x, bar_y + bar_h + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 1
            )

            cv2.imshow(win_name, combined)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            if key in SPEED_OPTIONS:
                speed_multiplier = SPEED_OPTIONS[key]

        cap_webcam.release()
        cap_ref.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _example_usage()
