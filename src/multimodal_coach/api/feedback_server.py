import json
import re
from typing import Any, Dict, List, Literal, Optional, Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from json_repair import repair_json

from ..pipelines.audio.event_analyzer import EventAnalyzerInput, run_rule_based_mvp


# ========= LM Studio 설정 =========
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen2.5-3b-instruct"

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")


# ========= 타입/스키마 =========
Metric = Literal["tempo", "pitch", "energy", "fluency", "emphasis", "pause"]
Score = Annotated[int, Field(ge=0, le=100)]


class SpeechScores(BaseModel):
    tempo: Score
    pitch: Score
    energy: Score
    fluency: Score
    emphasis: Score
    pause: Score

    # ===== Rule-based event analyzer input =====
    audio_duration: Optional[float] = None
    eval_duration: Optional[float] = None
    transcript: Optional[str] = None
    eval_gaps: Optional[List[float]] = None
    filler_counts: Optional[Dict[str, int]] = None

    word_timestamps: Optional[List[Dict[str, Any]]] = None
    silence_intervals: Optional[List[Dict[str, float]]] = None
    filler_occurrences: Optional[List[Dict[str, Any]]] = None


class TimestampedEvent(BaseModel):
    event_type: str
    start: float
    end: float
    severity: str
    score: float
    evidence: Dict[str, Any]
    feedback: str


class FeedbackCompact(BaseModel):
    total_mean: float
    priorities_top3: List[Metric]
    summary: str
    per_metric: Dict[Metric, str]
    note: Optional[str] = None

    event_overview: Optional[Dict[str, int]] = None
    timestamped_events: Optional[List[TimestampedEvent]] = None


# ========= 유틸리티 =========
def compute_total_mean(scores: Dict[str, int]) -> float:
    return round(sum(scores.values()) / 6, 2)


def pick_priorities(scores: Dict[str, int]) -> List[Metric]:
    weights = {
        "fluency": 1.15,
        "energy": 1.10,
        "tempo": 1.05,
        "pause": 1.00,
        "emphasis": 0.98,
        "pitch": 0.95,
    }
    ranked = sorted(scores.items(), key=lambda kv: (kv[1] * weights.get(kv[0], 1.0), kv[1]))
    return [ranked[0][0], ranked[1][0], ranked[2][0]]  # type: ignore


def extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output.")
    blob = m.group(0)
    try:
        return json.loads(repair_json(blob))
    except Exception:
        raise ValueError("JSON repair failed.")


def llm_json(system: str, user: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    text = resp.choices[0].message.content or ""
    return extract_json(text)


def coerce_per_metric(obj: Dict[str, Any]) -> Dict[Metric, str]:
    ordered: List[Metric] = ["tempo", "pitch", "energy", "fluency", "emphasis", "pause"]
    pm = obj.get("per_metric", {})
    if not isinstance(pm, dict):
        pm = {}

    out: Dict[Metric, str] = {}
    for k in ordered:
        v = pm.get(k)
        if not isinstance(v, str) or len(v.strip()) < 5:
            out[k] = "해당 지표에 대한 구체적인 보완이 필요해 보입니다. 기본기를 점검해 보세요."
        else:
            out[k] = v.strip()
    return out


def build_event_context(payload: SpeechScores) -> Dict[str, Any]:
    has_event_input = (
        payload.audio_duration is not None
        and payload.eval_duration is not None
        and payload.transcript is not None
        and payload.eval_gaps is not None
        and payload.filler_counts is not None
    )

    if not has_event_input:
        return {
            "event_overview": None,
            "timestamped_events": None,
            "event_text_for_prompt": "타임스탬프 기반 이벤트 정보는 제공되지 않았습니다."
        }

    analyzer_input = EventAnalyzerInput(
        audio_duration=payload.audio_duration,
        eval_duration=payload.eval_duration,
        transcript=payload.transcript,
        eval_gaps=payload.eval_gaps,
        filler_counts=payload.filler_counts,
        word_timestamps=payload.word_timestamps,
        silence_intervals=payload.silence_intervals,
        filler_occurrences=payload.filler_occurrences,
    )

    events_result = run_rule_based_mvp(analyzer_input)
    overview = events_result["event_overview"]
    events = events_result["timestamped_events"]

    if not events:
        event_text = "탐지된 타임스탬프 이벤트는 없습니다."
    else:
        lines = []
        for ev in events[:8]:
            lines.append(
                f"- {ev['event_type']} | {ev['start']}~{ev['end']}초 | "
                f"심각도={ev['severity']} | 근거={json.dumps(ev['evidence'], ensure_ascii=False)} | "
                f"해석={ev['feedback']}"
            )
        event_text = "탐지 이벤트 요약:\n" + "\n".join(lines)

    return {
        "event_overview": overview,
        "timestamped_events": events,
        "event_text_for_prompt": event_text,
    }


# ========= 프롬프트 및 로직 =========
SYSTEM = """
당신은 스피치 전문가입니다. 입력된 점수와 타임스탬프 이벤트를 바탕으로 사용자의 스피치 상태를 분석하고 개선 방향을 제시합니다.
반드시 한국어로 답변하며, 제공된 JSON 구조를 엄격히 지키세요.
항목별 피드백은 '현상(가능성) + 구체적 연습법'을 포함하여 1~2문장으로 작성합니다.

[절대 규칙]
1. 각 항목(per_metric)은 해당 지표만 집중해서 분석해라.
2. '연습하세요' 같은 추상 표현 대신 구체적 행동(Action Item)을 1개씩 포함해라.
3. 70점 이상은 강점으로 살리고, 60점 미만은 인지적 부하를 줄이는 연습법을 제안해라.
4. 타임스탬프 이벤트가 있으면 summary와 note에서 활용해도 되지만, per_metric에서 관련 없는 지표로 과도하게 번지지 마라.
5. timestamped event는 '근거'로만 사용하고, 없는 사실을 지어내지 마라.
"""


def build_user(
    scores: Dict[str, int],
    total_mean: float,
    priorities: List[Metric],
    event_text_for_prompt: str
) -> str:
    skeleton = {
        "summary": "입력된 점수와 이벤트를 종합 분석한 2문장",
        "per_metric": {
            "tempo": "이 항목에 대한 점수 기반 분석과 조언",
            "pitch": "이 항목에 대한 점수 기반 분석과 조언",
            "energy": "이 항목에 대한 점수 기반 분석과 조언",
            "fluency": "이 항목에 대한 점수 기반 분석과 조언",
            "emphasis": "이 항목에 대한 점수 기반 분석과 조언",
            "pause": "이 항목에 대한 점수 기반 분석과 조언"
        },
        "note": "분석의 한계점"
    }

    return (
        f"실제 데이터(점수): {json.dumps(scores, ensure_ascii=False)}\n"
        f"평균: {total_mean}, 우선개선: {priorities}\n"
        f"{event_text_for_prompt}\n\n"
        "필독 요구사항:\n"
        "1. skeleton에 적힌 문구를 그대로 출력하지 마세요.\n"
        "2. 반드시 실제 점수의 높고 낮음을 반영해 새로운 문장으로 작성하세요.\n"
        "3. 70점이 넘으면 강점으로, 60점 미만이면 구체적인 교정법으로 작성하세요.\n"
        "4. 이벤트 정보가 있으면 summary와 note에 반영할 수 있습니다.\n"
        "5. 반드시 아래 JSON 구조만 사용하세요.\n"
        f"{json.dumps(skeleton, ensure_ascii=False)}"
    )


# ========= FastAPI 서버 =========
app = FastAPI(title="Speech Coach AI (Event-aware)")


@app.post("/feedback", response_model=FeedbackCompact)
def feedback(payload: SpeechScores):
    scores = {
        "tempo": payload.tempo,
        "pitch": payload.pitch,
        "energy": payload.energy,
        "fluency": payload.fluency,
        "emphasis": payload.emphasis,
        "pause": payload.pause,
    }

    total_mean = compute_total_mean(scores)
    priorities = pick_priorities(scores)

    try:
        event_context = build_event_context(payload)

        raw = llm_json(
            system=SYSTEM,
            user=build_user(
                scores=scores,
                total_mean=total_mean,
                priorities=priorities,
                event_text_for_prompt=event_context["event_text_for_prompt"],
            )
        )

        out = {
            "total_mean": total_mean,
            "priorities_top3": priorities,
            "summary": str(raw.get("summary", "분석을 완료했습니다.")).strip(),
            "per_metric": coerce_per_metric(raw),
            "note": raw.get("note", "점수 기반 분석 결과입니다."),
            "event_overview": event_context["event_overview"],
            "timestamped_events": event_context["timestamped_events"],
        }
        return FeedbackCompact.model_validate(out)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback Generation Failed: {str(e)}")
