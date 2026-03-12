import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# =========================
# Data Schema
# =========================

@dataclass
class EventAnalyzerInput:
    audio_duration: float
    eval_duration: float
    transcript: str
    eval_gaps: List[float]
    filler_counts: Dict[str, int]

    # Optional richer inputs
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    silence_intervals: Optional[List[Dict[str, float]]] = None
    filler_occurrences: Optional[List[Dict[str, Any]]] = None


# =========================
# Utility
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def severity_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_korean_loose(text: str) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    return [t for t in text.split() if t]


def estimate_word_timestamps_from_transcript(
    transcript: str,
    eval_duration: float
) -> List[Dict[str, Any]]:
    tokens = tokenize_korean_loose(transcript)
    if not tokens:
        return []

    per = eval_duration / len(tokens)
    out = []
    cur = 0.0
    for tok in tokens:
        start = cur
        end = cur + per
        out.append({"word": tok, "start": start, "end": end})
        cur = end
    return out


def estimate_silence_intervals_from_gaps(
    eval_gaps: List[float],
    eval_duration: float
) -> List[Dict[str, float]]:
    if not eval_gaps:
        return []

    n = len(eval_gaps)
    anchors = [(i + 1) * eval_duration / (n + 1) for i in range(n)]

    intervals = []
    for a, g in zip(anchors, eval_gaps):
        start = max(0.0, a - g / 2)
        end = min(eval_duration, a + g / 2)
        intervals.append({"start": start, "end": end})
    return intervals


def summarize_event_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ev in events:
        et = ev["event_type"]
        out[et] = out.get(et, 0) + 1
    return out


# =========================
# Detector 1: Silence Event
# =========================

def detect_silence_events(
    data: EventAnalyzerInput,
    short_thresh: float = 0.8,
    long_thresh: float = 1.5
) -> List[Dict[str, Any]]:
    intervals = (
        data.silence_intervals
        if data.silence_intervals is not None
        else estimate_silence_intervals_from_gaps(data.eval_gaps, data.eval_duration)
    )

    events: List[Dict[str, Any]] = []
    for interval in intervals:
        start = float(interval["start"])
        end = float(interval["end"])
        dur = end - start

        if dur < short_thresh:
            continue

        if dur >= long_thresh:
            score = clamp(0.75 + 0.25 * (dur - long_thresh) / 1.5, 0.0, 1.0)
            feedback = "긴 침묵으로 인해 전달 흐름이 한 번 크게 끊겼습니다."
        else:
            score = clamp(
                0.4 + 0.35 * (dur - short_thresh) / max(long_thresh - short_thresh, 1e-6),
                0.0,
                1.0,
            )
            feedback = "침묵이 다소 길어 문장 연결이 잠깐 끊기는 인상을 줄 수 있습니다."

        events.append({
            "event_type": "silence_event",
            "start": round(start, 2),
            "end": round(end, 2),
            "severity": severity_from_score(score),
            "score": round(score, 3),
            "evidence": {
                "duration_sec": round(dur, 3),
                "short_threshold_sec": short_thresh,
                "long_threshold_sec": long_thresh,
            },
            "feedback": feedback,
        })

    return events


# =========================
# Detector 2: Filler Burst
# =========================

DEFAULT_FILLERS = {"어", "음", "그", "그러니까", "약간", "뭐랄까", "이제"}


def extract_fillers_from_words(
    words: List[Dict[str, Any]],
    filler_lexicon: set[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in words:
        token = str(w["word"]).strip()
        if token in filler_lexicon:
            out.append({
                "token": token,
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return out


def detect_filler_bursts(
    data: EventAnalyzerInput,
    filler_lexicon: Optional[set[str]] = None,
    window_sec: float = 10.0,
    min_count: int = 3,
    merge_gap_sec: float = 1.0
) -> List[Dict[str, Any]]:
    filler_lexicon = filler_lexicon or DEFAULT_FILLERS

    if data.filler_occurrences is not None:
        filler_occ = data.filler_occurrences
    else:
        words = data.word_timestamps
        if words is None:
            words = estimate_word_timestamps_from_transcript(data.transcript, data.eval_duration)
        filler_occ = extract_fillers_from_words(words, filler_lexicon)

    if not filler_occ:
        return []

    filler_occ = sorted(filler_occ, key=lambda x: x["start"])

    raw_windows = []
    n = len(filler_occ)
    j = 0
    for i in range(n):
        start_t = float(filler_occ[i]["start"])
        while j < n and float(filler_occ[j]["start"]) <= start_t + window_sec:
            j += 1
        count = j - i
        if count >= min_count:
            raw_windows.append({
                "start": start_t,
                "end": min(start_t + window_sec, data.eval_duration),
                "count": count,
                "tokens": [x["token"] for x in filler_occ[i:j]],
            })

    if not raw_windows:
        return []

    merged = []
    cur = raw_windows[0]
    for nxt in raw_windows[1:]:
        if float(nxt["start"]) <= float(cur["end"]) + merge_gap_sec:
            cur["end"] = max(float(cur["end"]), float(nxt["end"]))
            cur["count"] = max(int(cur["count"]), int(nxt["count"]))
            cur["tokens"].extend(nxt["tokens"])
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    events: List[Dict[str, Any]] = []
    for w in merged:
        count = int(w["count"])
        uniq_tokens = list(dict.fromkeys(w["tokens"]))
        score = clamp(0.35 + 0.15 * (count - min_count), 0.0, 1.0)

        events.append({
            "event_type": "filler_burst",
            "start": round(float(w["start"]), 2),
            "end": round(float(w["end"]), 2),
            "severity": severity_from_score(score),
            "score": round(score, 3),
            "evidence": {
                "window_sec": window_sec,
                "filler_count": count,
                "tokens": uniq_tokens[:10],
            },
            "feedback": "짧은 구간에 추임새가 몰려 나와 유창성이 약간 떨어질 수 있습니다.",
        })

    return events


# =========================
# Detector 3: Repair / Restart
# =========================

REPAIR_MARKERS = {
    "아니", "아니요", "아", "그러니까", "정정", "다시",
    "다시말해", "다시 말해", "정확히는", "정확히 말하면", "말하자면"
}


def detect_repeated_word_repairs(
    words: List[Dict[str, Any]],
    max_gap_sec: float = 0.7
) -> List[tuple[float, float, Dict[str, Any]]]:
    out = []
    for i in range(len(words) - 1):
        w1 = str(words[i]["word"]).strip()
        w2 = str(words[i + 1]["word"]).strip()
        if not w1 or not w2:
            continue
        if w1 == w2:
            gap = float(words[i + 1]["start"]) - float(words[i]["end"])
            if gap <= max_gap_sec:
                out.append((
                    float(words[i]["start"]),
                    float(words[i + 1]["end"]),
                    {
                        "pattern": "repetition",
                        "token": w1,
                        "gap_sec": round(gap, 3),
                    }
                ))
    return out


def detect_marker_repairs(
    words: List[Dict[str, Any]],
    repair_markers: set[str]
) -> List[tuple[float, float, Dict[str, Any]]]:
    out = []
    joined_words = [str(w["word"]).strip() for w in words]

    for i, tok in enumerate(joined_words):
        if tok in repair_markers:
            out.append((
                float(words[i]["start"]),
                float(words[i]["end"]),
                {
                    "pattern": "repair_marker",
                    "token": tok,
                }
            ))

    for i in range(len(joined_words) - 1):
        phrase = joined_words[i] + " " + joined_words[i + 1]
        if phrase in repair_markers:
            out.append((
                float(words[i]["start"]),
                float(words[i + 1]["end"]),
                {
                    "pattern": "repair_marker_phrase",
                    "token": phrase,
                }
            ))
    return out


def detect_repair_restart(
    data: EventAnalyzerInput,
    repair_markers: Optional[set[str]] = None
) -> List[Dict[str, Any]]:
    repair_markers = repair_markers or REPAIR_MARKERS

    words = data.word_timestamps
    if words is None:
        words = estimate_word_timestamps_from_transcript(data.transcript, data.eval_duration)

    if not words:
        return []

    candidates: List[tuple[float, float, Dict[str, Any]]] = []
    candidates.extend(detect_repeated_word_repairs(words))
    candidates.extend(detect_marker_repairs(words, repair_markers))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])

    merged = []
    cur_s, cur_e, cur_ev = candidates[0]
    cur_evidence = [cur_ev]

    for s, e, ev in candidates[1:]:
        if s <= cur_e + 0.8:
            cur_e = max(cur_e, e)
            cur_evidence.append(ev)
        else:
            merged.append((cur_s, cur_e, cur_evidence))
            cur_s, cur_e, cur_ev = s, e, ev
            cur_evidence = [cur_ev]
    merged.append((cur_s, cur_e, cur_evidence))

    events: List[Dict[str, Any]] = []
    for s, e, evidence_list in merged:
        patterns = [ev["pattern"] for ev in evidence_list]
        tokens = [ev["token"] for ev in evidence_list if "token" in ev]
        count = len(evidence_list)
        score = clamp(0.35 + 0.2 * (count - 1), 0.0, 1.0)

        if "repetition" in patterns and any("repair_marker" in p for p in patterns):
            feedback = "표현을 반복한 뒤 수정하는 흐름이 보여 발화가 잠깐 흔들렸습니다."
        elif "repetition" in patterns:
            feedback = "같은 단어를 반복하며 시작이 잠깐 흔들렸습니다."
        else:
            feedback = "표현을 고쳐 말하는 구간이 있어 전달 흐름이 잠시 끊겼습니다."

        events.append({
            "event_type": "repair_restart",
            "start": round(s, 2),
            "end": round(e, 2),
            "severity": severity_from_score(score),
            "score": round(score, 3),
            "evidence": {
                "count": count,
                "patterns": patterns,
                "tokens": tokens,
            },
            "feedback": feedback,
        })

    return events


# =========================
# Main Runner
# =========================

def run_rule_based_mvp(data: EventAnalyzerInput) -> Dict[str, Any]:
    silence_events = detect_silence_events(data)
    filler_events = detect_filler_bursts(data)
    repair_events = detect_repair_restart(data)

    all_events = sorted(
        silence_events + filler_events + repair_events,
        key=lambda x: (x["start"], -x["score"])
    )

    return {
        "event_overview": summarize_event_counts(all_events),
        "timestamped_events": all_events,
    }