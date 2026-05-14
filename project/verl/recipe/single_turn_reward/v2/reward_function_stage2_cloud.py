from __future__ import annotations

from typing import Any

from recipe.single_turn_reward.v2.reward_model_stage2_cloud import compute_model_style_score


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


def _expected_stage_hint(extra_info: dict[str, Any] | None) -> str:
    extra_info = extra_info or {}
    turn_round = int(extra_info.get("turn_round") or 0)
    contact_round = int(extra_info.get("rule_contact_round") or 0)
    # First two turns are always treated as start-stage in this task.
    if turn_round <= 2:
        return "start"
    if contact_round > 0 and turn_round >= contact_round:
        return "contact"
    return "mid"


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

    enable_model_judge = bool(kwargs.get("enable_model_judge", True))
    if enable_model_judge:
        history_text = _history_text(extra_info)
        stage_hint = _expected_stage_hint(extra_info)
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
            expected_stage_hint=stage_hint,
            stage_weight=float(kwargs.get("stage_weight", 0.25)),
            objective_weight=float(kwargs.get("objective_weight", 0.35)),
            professional_weight=float(kwargs.get("professional_weight", 0.15)),
            safety_weight=float(kwargs.get("safety_weight", 0.10)),
            benchmark_weight=float(kwargs.get("benchmark_weight", 0.15)),
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
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }
        model_score = 0.5

    score = _clip(model_score)

    if bool(model_res.get("redline", False)):
        score = min(score, 0.2)

    return {
        "score": score,
        "rule_score": 0.0,
        "model_judge_score": _clip(model_score),
        "model_judge_status": str(model_res.get("status", "")),
        "model_stage": str(model_res.get("stage", "unknown")),
        "model_stage_fit_score": _clip(float(model_res.get("stage_fit_score", 0.5))),
        "model_objective_score": _clip(float(model_res.get("objective_score", 0.5))),
        "model_professional_score": _clip(float(model_res.get("professional_score", 0.5))),
        "model_safety_score": _clip(float(model_res.get("safety_score", 0.5))),
        "model_benchmark_alignment_score": _clip(float(model_res.get("benchmark_alignment_score", 0.5))),
        "model_style_penalty": _clip(float(model_res.get("style_penalty", 0.0)), 0.0, 1.0),
        "model_template_tone": bool(model_res.get("template_tone", False)),
        "model_no_progress": bool(model_res.get("no_progress", False)),
        "rule_weight": 0.0,
        "model_weight": 1.0,
    }
