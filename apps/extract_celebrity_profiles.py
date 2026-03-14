"""
유명인 영상 프로필 추출 실행 스크립트

사용법:
    # 전체 data/ 폴더 처리
    PYTHONPATH=src python apps/extract_celebrity_profiles.py

    # 특정 영상만 처리
    PYTHONPATH=src python apps/extract_celebrity_profiles.py --video data/obama1.mp4

결과:
    assets/celebrity_profiles/{name}.profile.json 으로 저장됨
    각 파일에 vision, audio 수치와 LLM용 프롬프트(prompt 필드) 포함
"""

import argparse
import sys
from pathlib import Path

# PYTHONPATH=src 없이도 동작하도록
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_coach.pipelines.celebrity_profiler import CelebrityProfiler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "celebrity_profiles"


def main():
    parser = argparse.ArgumentParser(description="유명인 영상 프로필 추출")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="특정 영상 경로 (미지정 시 data/ 폴더 전체 처리)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiler = CelebrityProfiler()

    if args.video:
        video_files = [Path(args.video)]
    else:
        video_files = sorted(DATA_DIR.glob("*.mp4"))

    if not video_files:
        print(f"처리할 mp4 파일이 없습니다: {DATA_DIR}")
        return

    print(f"처리할 영상 {len(video_files)}개: {[v.name for v in video_files]}\n")

    for video_path in video_files:
        print(f"{'='*50}")
        print(f"처리 중: {video_path.name}")
        print(f"{'='*50}")
        try:
            profile = profiler.extract(
                video_path=video_path,
                output_path=OUTPUT_DIR / f"{video_path.stem}.profile.json",
            )
            print("\n[요약]")
            print(profile["summary"])
            print("\n[LLM 프롬프트 미리보기]")
            print(profile["prompt"])
            print()
        except Exception as e:
            print(f"오류 ({video_path.name}): {e}\n")

    print(f"\n완료. 결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
