"""
유명인 영상 분석 → 텍스트 프롬프트 추출 모듈

사용법:
    profiler = CelebrityProfiler()
    profile = profiler.extract("data/obama1.mp4")
    # → assets/celebrity_profiles/obama1.profile.json 저장
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

from .vision.pose_analyzer import PoseAnalyzer, AlertChecker, LEARNED_PUNCH_PATH
from .vision.gaze import GazeAnxietyDetector


class CelebrityProfiler:
    """
    유명인 영상에서 자세/시선/음성 특성을 추출하여 프로필 JSON과 텍스트 프롬프트 저장.

    추출 항목:
        Vision:
            - avg_body_tilt_deg: 평균 몸 기울기 (도)
            - avg_tremor_level:  평균 떨림 수준 (0~1)
            - gaze_stable_ratio / gaze_avoiding_ratio / gaze_shaking_ratio
            - alert_counts: 자세 경고 발생 횟수 (body_tilt, head_tilt, body_tremor)
        Audio:
            - avg_energy:    평균 RMS 에너지
            - avg_pitch_hz:  평균 기본 주파수 (Hz)
            - pitch_std_hz:  피치 표준편차 (억양 폭)
            - voiced_ratio:  유성음 구간 비율 (말하는 시간 비율)
            - duration_sec:  영상 길이 (초)
    """

    def __init__(self):
        self.pose_analyzer = PoseAnalyzer(learned_punch_path=LEARNED_PUNCH_PATH)
        self.alert_checker = AlertChecker(self.pose_analyzer)

    def extract(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> dict:
        """
        영상 분석 후 프로필 dict 반환 및 JSON 저장.

        Args:
            video_path:  분석할 mp4 경로
            output_path: 저장 경로 (미지정 시 video_path 옆에 .profile.json 으로 저장)

        Returns:
            {celebrity, source_video, vision, audio, summary, prompt} 형태의 dict
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")

        print(f"[Vision] 분석 중: {video_path.name}")
        vision_metrics = self._analyze_vision(video_path)

        print(f"[Audio]  분석 중: {video_path.name}")
        audio_metrics = self._analyze_audio(video_path)

        celebrity_name = video_path.stem  # "obama1"
        summary = self._build_summary(celebrity_name, vision_metrics, audio_metrics)
        prompt = self._build_llm_prompt(celebrity_name, vision_metrics, audio_metrics)

        profile = {
            "celebrity": celebrity_name,
            "source_video": str(video_path),
            "vision": vision_metrics,
            "audio": audio_metrics,
            "summary": summary,
            "prompt": prompt,
        }

        if output_path is None:
            output_path = video_path.with_suffix(".profile.json")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        print(f"[저장]   {output_path}")
        return profile

    # -------------------------------------------------------------------------
    # Vision 분석
    # -------------------------------------------------------------------------

    def _analyze_vision(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 첫 1/4 구간을 시선 캘리브레이션으로 사용 (최소 30, 최대 90 프레임)
        calib_frames = max(30, min(90, total_frames // 4))
        gaze_detector = GazeAnxietyDetector(calibration_frames=calib_frames)

        mp_holistic = mp.solutions.holistic

        body_tilts: list[float] = []
        tremor_levels: list[float] = []
        gaze_counts: dict[str, int] = defaultdict(int)
        alert_counts: dict[str, int] = defaultdict(int)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                # --- 자세 분석 ---
                metrics = self.pose_analyzer.analyze(
                    results.pose_landmarks,
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                )
                body_tilts.append(metrics.body_tilt_angle)
                tremor_levels.append(metrics.tremor_level)

                alerts = self.alert_checker.check_alerts(metrics)
                for alert in alerts:
                    alert_counts[alert.alert_type.value] += 1

                # --- 시선 분석 ---
                gaze_result = gaze_detector.process_frame(frame)
                gaze_counts[gaze_result.get("status", "Unknown")] += 1

        cap.release()
        gaze_detector.release()

        total = max(len(body_tilts), 1)
        return {
            "total_frames_analyzed": total,
            "avg_body_tilt_deg": round(float(np.mean(body_tilts)), 2) if body_tilts else 0.0,
            "max_body_tilt_deg": round(float(np.max(body_tilts)), 2) if body_tilts else 0.0,
            "avg_tremor_level": round(float(np.mean(tremor_levels)), 3) if tremor_levels else 0.0,
            "gaze_stable_ratio": round(gaze_counts.get("Stable", 0) / total, 3),
            "gaze_avoiding_ratio": round(gaze_counts.get("Avoiding", 0) / total, 3),
            "gaze_shaking_ratio": round(gaze_counts.get("Shaking", 0) / total, 3),
            "alert_counts": dict(alert_counts),
        }

    # -------------------------------------------------------------------------
    # Audio 분석
    # -------------------------------------------------------------------------

    def _analyze_audio(self, video_path: Path) -> dict:
        if not _HAS_LIBROSA:
            return {"error": "librosa not installed"}

        try:
            import av as pyav

            # PyAV로 mp4에서 오디오 샘플 직접 추출 (내장 ffmpeg 사용, 시스템 ffmpeg 불필요)
            container = pyav.open(str(video_path))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                return {"error": "오디오 트랙 없음"}

            sr = audio_stream.sample_rate
            chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()          # shape: (channels, samples)
                chunks.append(arr.mean(axis=0))   # mono 변환
            container.close()

            if not chunks:
                return {"error": "오디오 샘플 없음"}

            y = np.concatenate(chunks).astype(np.float32)
            # PyAV는 int16 범위로 출력 → -1~1 정규화
            if y.dtype != np.float32 or y.max() > 1.0:
                y = y / 32768.0

            # 에너지 (RMS)
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = float(np.mean(rms))

            # 피치 (기본 주파수, pyin 알고리즘)
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            voiced_f0 = f0[voiced_flag] if (f0 is not None and voiced_flag is not None) else np.array([])
            avg_pitch_hz = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            pitch_std_hz = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            voiced_ratio = (
                float(np.sum(voiced_flag) / len(voiced_flag))
                if voiced_flag is not None and len(voiced_flag) > 0
                else 0.0
            )

            return {
                "avg_energy": round(avg_energy, 4),
                "avg_pitch_hz": round(avg_pitch_hz, 1),
                "pitch_std_hz": round(pitch_std_hz, 1),
                "voiced_ratio": round(voiced_ratio, 3),
                "duration_sec": round(float(len(y) / sr), 1),
            }
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # 프롬프트 생성
    # -------------------------------------------------------------------------

    def _build_summary(self, name: str, vision: dict, audio: dict) -> str:
        """수치 기반 자연어 요약 (사람이 읽는 용도)"""
        lines = [f"[{name}] 발표 스타일 분석 요약"]

        # 자세
        tilt = vision.get("avg_body_tilt_deg", 0.0)
        tremor = vision.get("avg_tremor_level", 0.0)
        if tilt < 5:
            lines.append(f"- 자세: 매우 바른 자세 유지 (평균 기울기 {tilt:.1f}°)")
        elif tilt < 10:
            lines.append(f"- 자세: 대체로 바른 자세 (평균 기울기 {tilt:.1f}°)")
        else:
            lines.append(f"- 자세: 눈에 띄는 몸 기울기 (평균 {tilt:.1f}°)")

        if tremor < 0.1:
            lines.append("- 움직임: 차분하고 안정적")
        elif tremor < 0.4:
            lines.append(f"- 움직임: 약간의 몸 움직임 (떨림 수준 {tremor:.2f})")
        else:
            lines.append(f"- 움직임: 눈에 띄는 몸 떨림 (수준 {tremor:.2f})")

        # 시선
        stable = vision.get("gaze_stable_ratio", 0.0)
        avoiding = vision.get("gaze_avoiding_ratio", 0.0)
        if stable > 0.75:
            lines.append(f"- 시선: 안정적인 아이컨택 ({stable*100:.0f}% Stable)")
        elif stable > 0.5:
            lines.append(f"- 시선: 보통 수준 아이컨택 ({stable*100:.0f}% Stable)")
        else:
            lines.append(f"- 시선: 시선 회피 빈번 ({avoiding*100:.0f}% Avoiding)")

        # 음성
        energy = audio.get("avg_energy", 0.0)
        pitch_std = audio.get("pitch_std_hz", 0.0)
        voiced_ratio = audio.get("voiced_ratio", 0.0)

        if energy > 0.05:
            lines.append("- 음량: 크고 힘찬 발성")
        elif energy > 0.02:
            lines.append("- 음량: 적당한 발성 에너지")
        else:
            lines.append("- 음량: 작은 발성 에너지")

        if pitch_std > 40:
            lines.append(f"- 억양: 풍부한 음정 변화 (피치 표준편차 {pitch_std:.0f}Hz)")
        elif pitch_std > 20:
            lines.append(f"- 억양: 적당한 음정 변화 ({pitch_std:.0f}Hz)")
        else:
            lines.append(f"- 억양: 단조로운 음조 ({pitch_std:.0f}Hz)")

        if voiced_ratio > 0.7:
            lines.append("- 유창성: 유창하게 말함 (침묵 적음)")
        else:
            lines.append(f"- 유창성: 침묵 구간 많음 (유성음 비율 {voiced_ratio:.0%})")

        return "\n".join(lines)

    def _build_llm_prompt(self, name: str, vision: dict, audio: dict) -> str:
        """
        실시간 평가 시 LLM에 넘길 레퍼런스 프롬프트.
        사용자 수치와 함께 이 텍스트를 LLM에 제공하여 비교 피드백을 생성.
        """
        return (
            f"Reference speaker: {name}\n"
            f"Posture: avg body tilt {vision.get('avg_body_tilt_deg', 0):.1f}°, "
            f"tremor level {vision.get('avg_tremor_level', 0):.2f}.\n"
            f"Gaze: stable {vision.get('gaze_stable_ratio', 0)*100:.0f}%, "
            f"avoiding {vision.get('gaze_avoiding_ratio', 0)*100:.0f}%.\n"
            f"Voice: energy {audio.get('avg_energy', 0):.4f}, "
            f"pitch std {audio.get('pitch_std_hz', 0):.0f}Hz, "
            f"voiced ratio {audio.get('voiced_ratio', 0):.0%}.\n"
            f"Use these reference values to evaluate and give feedback to the user "
            f"who is trying to emulate this speaking style."
        )
