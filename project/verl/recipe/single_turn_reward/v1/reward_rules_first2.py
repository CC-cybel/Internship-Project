from __future__ import annotations

import re
from typing import Any

_FORMAT_RE = re.compile(
    r"^BEGIN_META\n(?P<meta>[\s\S]*?)\nEND_META\nBEGIN_FINAL\n(?P<final>[\s\S]*?)\nEND_FINAL$",
    flags=re.DOTALL,
)
_CONTACT_RE = re.compile(r"(联系方式|手机号|电话|微信|vx|留个|联系你|联系您)", flags=re.IGNORECASE)
_HARD_SELL_RE = re.compile(r"(必须|立刻|马上|赶紧|不留就|现在就留|抓紧留)")
_REDLINE_RE = re.compile(r"(不留就|吓唬|威胁|你必须|不然会)")


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _keywords(text: str) -> set[str]:
    toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text or "")
    stop = {"当前", "系统", "数据", "继续", "对话", "你好", "请问", "我们", "你们"}
    return {t.lower() for t in toks if t.lower() not in stop}


def _extract(answer: str) -> tuple[list[str], str]:
    m = _FORMAT_RE.match((answer or "").strip())
    if not m:
        return [], ""
    meta_lines = [ln for ln in m.group("meta").split("\n") if ln.strip()]
    final_text = m.group("final").strip()
    return meta_lines, final_text


def _infer_expected_stage(extra_info: dict[str, Any]) -> str:
    turn_round = int(extra_info.get("turn_round") or 0)
    contact_round = int(extra_info.get("rule_contact_round") or 0)

    if contact_round > 0 and turn_round >= contact_round:
        return "contact"
    if turn_round <= 1:
        return "start"
    if turn_round > 1:
        return "mid"
    return "mid"


def _stage_fit_score(expected_stage: str, text: str, asks_contact: bool, qmarks: int) -> float:
    if expected_stage == "start":
        if asks_contact:
            return 0.25
        if 1 <= qmarks <= 3:
            return 1.0
        return 0.7
    if expected_stage == "mid":
        if asks_contact:
            return 0.6
        return 0.9 if len(text) >= 35 else 0.7
    if expected_stage == "contact":
        return 1.0 if asks_contact else 0.45
    return 0.7


def _objective_score(expected_stage: str, text: str, asks_contact: bool, qmarks: int) -> float:
    diagnostic_hits = sum(1 for p in ("多久", "频率", "症状", "影响", "什么时候", "程度") if p in text)
    value_hits = sum(1 for p in ("建议", "方案", "分析", "评估", "步骤", "先", "再") if p in text)
    exchange_hits = sum(1 for p in ("给你", "发你", "资料", "方案", "不打扰", "方便") if p in text)

    if expected_stage == "start":
        return _clip(0.3 + 0.25 * min(2, diagnostic_hits) + (0.2 if 1 <= qmarks <= 3 else 0.0))
    if expected_stage == "mid":
        return _clip(0.3 + 0.3 * min(2, value_hits) + (0.1 if len(text) >= 40 else 0.0))
    if expected_stage == "contact":
        return _clip((0.55 if asks_contact else 0.2) + 0.2 * min(2, exchange_hits))
    return 0.5


def _professional_score(text: str, relevance_score: float, empathy_score: float) -> float:
    structure_hits = sum(1 for p in ("先", "然后", "最后", "建议", "可以") if p in text)
    return _clip(0.35 * relevance_score + 0.35 * empathy_score + 0.30 * _clip(structure_hits / 3))


def _safety_score(text: str, asks_contact: bool, expected_stage: str) -> tuple[float, bool, bool, bool]:
    hard_sell = bool(_HARD_SELL_RE.search(text))
    redline = bool(_REDLINE_RE.search(text))
    premature_contact = expected_stage == "start" and asks_contact

    score = 1.0
    if premature_contact:
        score -= 0.4
    if hard_sell:
        score -= 0.4
    if redline:
        score -= 0.8
    return _clip(score), premature_contact, hard_sell, redline


def compute_rule_components(question: str, answer: str, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    extra_info = extra_info or {}
    meta_lines, final_text = _extract(answer)

    markers = all(tag in (answer or "") for tag in ("BEGIN_META", "END_META", "BEGIN_FINAL", "END_FINAL"))
    format_score = 1.0 if markers and final_text else 0.0
    if meta_lines:
        if not meta_lines[0].startswith("action="):
            format_score *= 0.6
        if len(meta_lines) < 2 or not meta_lines[1].startswith("thought="):
            format_score *= 0.7

    qk = _keywords(question)
    ak = _keywords(final_text)
    overlap = len(qk & ak)
    relevance_score = 0.0 if not qk else _clip(overlap / max(1, min(len(qk), 6)))
    if len(final_text) < 20:
        relevance_score *= 0.6

    empathy_hits = sum(1 for p in ("理解", "明白", "不容易", "别担心", "我在", "我们一起", "我会帮") if p in final_text)
    empathy_score = _clip(0.2 + 0.4 * min(2, empathy_hits))

    qmarks = final_text.count("?") + final_text.count("？")
    progress_score = 0.2 if qmarks == 0 else (1.0 if qmarks <= 2 else 0.6)
    if len(final_text) >= 40:
        progress_score = _clip(progress_score + 0.1)

    turn_round = int(extra_info.get("turn_round") or 0)
    contact_round = int(extra_info.get("rule_contact_round") or 0)
    asks_contact = bool(_CONTACT_RE.search(final_text))
    if contact_round <= 0 or turn_round <= 0:
        contact_policy = 0.7
    elif turn_round < contact_round:
        contact_policy = 1.0 if not asks_contact else 0.3
    else:
        contact_policy = 1.0 if asks_contact else 0.5

    expected_stage = _infer_expected_stage(extra_info)
    stage_fit = _stage_fit_score(expected_stage=expected_stage, text=final_text, asks_contact=asks_contact, qmarks=qmarks)
    objective = _objective_score(expected_stage=expected_stage, text=final_text, asks_contact=asks_contact, qmarks=qmarks)
    professional = _professional_score(final_text, relevance_score=relevance_score, empathy_score=empathy_score)
    safety, premature_contact, hard_sell, redline = _safety_score(
        text=final_text,
        asks_contact=asks_contact,
        expected_stage=expected_stage,
    )

    total = (
        0.10 * format_score
        + 0.10 * relevance_score
        + 0.10 * empathy_score
        + 0.05 * progress_score
        + 0.05 * contact_policy
        + 0.25 * stage_fit
        + 0.20 * objective
        + 0.10 * professional
        + 0.05 * safety
    )

    if redline:
        total = min(total, 0.2)

    return {
        "score": _clip(total),
        "format_score": _clip(format_score),
        "relevance_score": _clip(relevance_score),
        "empathy_score": _clip(empathy_score),
        "progress_score": _clip(progress_score),
        "contact_policy_score": _clip(contact_policy),
        "turn_round": turn_round,
        "contact_round": contact_round,
        "asks_contact": asks_contact,
        "expected_stage": expected_stage,
        "stage_fit_score": _clip(stage_fit),
        "objective_score": _clip(objective),
        "professional_score": _clip(professional),
        "safety_score": _clip(safety),
        "premature_contact": premature_contact,
        "hard_sell": hard_sell,
        "redline": redline,
    }
