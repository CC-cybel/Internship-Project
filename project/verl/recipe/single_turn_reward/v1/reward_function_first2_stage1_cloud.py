from __future__ import annotations

from typing import Any

from recipe.single_turn_reward.v1.reward_model_first2_cloud import compute_model_style_score
from recipe.single_turn_reward.v1.reward_rules_first2 import compute_rule_components


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _question(extra_info: dict[str, Any] | None, ground_truth: str) -> str:
    extra_info = extra_info or {}
    for key in ("question", "instruction", "query"):
        v = extra_info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return (ground_truth or "").strip()


def _normalize_turn(turn: Any) -> tuple[str, str] | None:
    if not isinstance(turn, dict):
        return None

    role = turn.get("role")
    content = turn.get("content")
    if role is None or content is None:
        role = turn.get("from")
        content = turn.get("value")

    if role is None or content is None:
        return None

    role = str(role).strip().lower()
    if role in {"human", "user"}:
        role = "user"
    elif role in {"assistant", "bot", "gpt"}:
        role = "assistant"
    else:
        return None

    text = str(content).strip()
    if not text:
        return None
    return role, text


def _history_text(extra_info: dict[str, Any] | None, max_turns: int = 8) -> str:
    extra_info = extra_info or {}
    turns = None
    for key in ("history", "conversations", "dialogue"):
        value = extra_info.get(key)
        if isinstance(value, list):
            turns = value
            break
    if not turns:
        return ""

    normalized: list[tuple[str, str]] = []
    for t in turns:
        item = _normalize_turn(t)
        if item is not None:
            normalized.append(item)
    if not normalized:
        return ""

    clipped = normalized[-max_turns:]
    lines = [f"{r}: {c}" for r, c in clipped]
    return "\n".join(lines)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del data_source
    del reward_router_address
    del reward_model_tokenizer

    q = _question(extra_info, ground_truth)
    a = solution_str or ""

    rule = compute_rule_components(question=q, answer=a, extra_info=extra_info)
    rule_score = float(rule.get("score", 0.0))

    enable_model_judge = bool(kwargs.get("enable_model_judge", True))
    if enable_model_judge:
        history_text = _history_text(extra_info)
        model_res = await compute_model_style_score(
            question=q,
            answer=a,
            api_base=str(kwargs.get("api_base", "")),
            api_key=str(kwargs.get("api_key", "")),
            judge_model=str(kwargs.get("judge_model", "")),
            timeout_s=float(kwargs.get("model_judge_timeout_s", 45.0)),
            history_text=history_text,
            turn_round=int((extra_info or {}).get("turn_round") or 0),
            contact_round=int((extra_info or {}).get("rule_contact_round") or 0),
            expected_stage_hint=str(rule.get("expected_stage", "")),
        )
        model_score = float(model_res.get("score", 0.5))
    else:
        model_res = {
            "score": 0.5,
            "status": "disabled",
            "raw": "",
            "stage": "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
        }
        model_score = 0.5

    rw = float(kwargs.get("rule_weight", 0.7))
    mw = float(kwargs.get("model_weight", 0.3))
    rw = max(0.0, rw)
    mw = max(0.0, mw)
    s = rw + mw
    if s <= 0:
        rw, mw, s = 1.0, 0.0, 1.0

    score = _clip((rw * rule_score + mw * model_score) / s)

    # Light agreement bonus/penalty between deterministic stage expectation and LLM stage prediction.
    rule_stage = str(rule.get("expected_stage", ""))
    model_stage = str(model_res.get("stage", ""))
    if rule_stage in {"start", "mid", "contact"} and model_stage in {"start", "mid", "contact"}:
        if rule_stage == model_stage:
            score = _clip(score + 0.02)
        else:
            score = _clip(score - 0.02)

    return {
        "score": score,
        "rule_score": _clip(rule_score),
        "model_judge_score": _clip(model_score),
        "model_judge_status": str(model_res.get("status", "")),
        "model_stage": str(model_res.get("stage", "unknown")),
        "model_stage_fit_score": _clip(float(model_res.get("stage_fit_score", 0.5))),
        "model_objective_score": _clip(float(model_res.get("objective_score", 0.5))),
        "model_professional_score": _clip(float(model_res.get("professional_score", 0.5))),
        "model_safety_score": _clip(float(model_res.get("safety_score", 0.5))),
        "rule_weight": rw,
        "model_weight": mw,
        **{k: v for k, v in rule.items() if k != "score"},
    }
