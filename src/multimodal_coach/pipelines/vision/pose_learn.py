"""
면접 피드백 - 포즈 학습 모듈

주먹 날리기 등 특정 자세를 학습하여 Test2_MP에서 사용할 수 있습니다.
- 's' 키: 현재 포즈를 "주먹" 샘플로 저장
- 'q' 키: 종료 및 저장
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Test2_MP와 공유하는 랜드마크 인덱스
POSE_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / "punch_pose_samples.json"


def landmarks_to_feature(pose_landmarks, left_hand, right_hand) -> dict | None:
    """
    Holistic 결과에서 주먹 자세 특징 벡터 추출.
    어깨, 팔꿈치, 손목 + 손 랜드마크 (있을 경우)
    """
    if pose_landmarks is None:
        return None
    lm = pose_landmarks.landmark

    feature = {"pose": [], "left_hand": None, "right_hand": None}

    for name, idx in POSE_INDICES.items():
        p = lm[idx]
        feature["pose"].append({"x": p.x, "y": p.y, "z": p.z})

    def hand_to_list(hand):
        if hand is None:
            return None
        return [
            {"x": p.x, "y": p.y, "z": p.z}
            for p in hand.landmark
        ]

    feature["left_hand"] = hand_to_list(left_hand)
    feature["right_hand"] = hand_to_list(right_hand)
    return feature


def load_samples(path: Path) -> list[dict]:
    """저장된 샘플 로드"""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


def save_samples(samples: list[dict], path: Path) -> None:
    """샘플 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gesture": "punch", "samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {path} ({len(samples)}개 샘플)")


def run_learn_punch(save_path: Path = DEFAULT_SAVE_PATH):
    """
    주먹 자세 학습 모드.
    - 웹캠으로 포즈 캡처
    - 's': 현재 포즈 저장 (주먹 자세 취한 후)
    - 'q': 종료 및 저장
    """
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    samples = load_samples(save_path)
    print(f"기존 샘플 {len(samples)}개 로드. 새로 저장 시 추가됩니다.")
    print("주먹 자세를 취하고 's' 키로 저장, 'q' 키로 종료")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 찾을 수 없습니다.")
        return

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

            # 랜드마크 그리기
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

            # UI
            cv2.putText(image, "[s] Save punch pose  [q] Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Samples: {len(samples)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Pose Learn - Punch", image)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                feat = landmarks_to_feature(
                    results.pose_landmarks,
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                )
                if feat:
                    samples.append(feat)
                    print(f"샘플 저장 ({len(samples)}번째)")
                else:
                    print("포즈를 인식하지 못했습니다. 카메라에 몸이 보이도록 해주세요.")

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        save_samples(samples, save_path)
    else:
        print("저장된 샘플이 없습니다.")


def compute_similarity(feat_a: dict, feat_b: dict) -> float:
    """
    두 특징 벡터 간 유사도 (0~1, 1=동일).
    pose 벡터만 사용하여 L2 거리 기반 유사도 반환.
    """
    pa = feat_a.get("pose", [])
    pb = feat_b.get("pose", [])
    if len(pa) != len(pb) or len(pa) == 0:
        return 0.0
    va = np.array([[p["x"], p["y"], p["z"]] for p in pa]).flatten()
    vb = np.array([[p["x"], p["y"], p["z"]] for p in pb]).flatten()
    dist = np.linalg.norm(va - vb)
    # 거리 → 유사도: dist 0.5 이상이면 낮은 유사도
    sigma = 0.3
    similarity = np.exp(-dist**2 / (2 * sigma**2))
    return float(similarity)


def is_punch_from_learned(
    pose_landmarks,
    left_hand,
    right_hand,
    samples: list[dict],
    threshold: float = 0.7,
) -> bool:
    """
    학습된 샘플과 비교하여 주먹 자세인지 판별.
    samples가 비어 있으면 False (규칙 기반 fallback은 Test2_MP에서 처리).
    """
    if not samples:
        return False
    current = landmarks_to_feature(pose_landmarks, left_hand, right_hand)
    if current is None:
        return False
    for s in samples:
        sim = compute_similarity(current, s)
        if sim >= threshold:
            return True
    return False


if __name__ == "__main__":
    run_learn_punch()
