"""
영상에서 Whisper를 사용해 실제 대사를 추출하고,
_subs.json 형식으로 저장하는 스크립트.

faster-whisper 사용 (openai-whisper 대비 가볍고 numba 불필요).

설치:
  pip install faster-whisper
  # ffmpeg 필요: brew install ffmpeg (macOS)

실행:
  python extract_subtitles.py [영상경로]
  python extract_subtitles.py "Obama's 2004 DNC keynote speech.mp4"
  python extract_subtitles.py video.mp4 -m medium -o my_subs.json
"""

# OpenMP libiomp5 중복 로드 오류 방지 (macOS 등)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path


def extract_subtitles_with_whisper(
    video_path: str | Path,
    model_name: str = "base",
) -> list[dict]:
    """
    faster-whisper로 영상 음성 인식 후, start_sec/end_sec/text 형식으로 변환.

    Returns:
        [{"start_sec": float, "end_sec": float, "text": str}, ...]
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper가 설치되어 있지 않습니다. 다음을 실행하세요:\n"
            "  pip install faster-whisper\n"
            "ffmpeg도 필요합니다: brew install ffmpeg (macOS)"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")

    print(f"Whisper 모델 로딩: {model_name} (첫 실행 시 다운로드됨)")
    model = WhisperModel(model_name)

    print(f"영상 음성 인식 중: {video_path.name}")
    segments_iter, _ = model.transcribe(str(video_path), language="en")

    segments = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            segments.append({
                "start_sec": round(seg.start, 2),
                "end_sec": round(seg.end, 2),
                "text": text,
            })

    return segments


def main():
    parser = argparse.ArgumentParser(description="영상에서 Whisper로 자막 추출")
    parser.add_argument(
        "video",
        nargs="?",
        default="Obama's 2004 DNC keynote speech.mp4",
        help="입력 영상 경로 (기본: Obama's 2004 DNC keynote speech.mp4)",
    )
    parser.add_argument(
        "-o", "--output",
        help="출력 JSON 경로 (미지정 시 영상이름_subs.json)",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 모델 (tiny=최소, base=빠름, medium=정확)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output) if args.output else video_path.with_name(
        video_path.stem + "_subs.json"
    )

    segments = extract_subtitles_with_whisper(video_path, model_name=args.model)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"완료: {len(segments)}개 구간 → {output_path}")


if __name__ == "__main__":
    main()
