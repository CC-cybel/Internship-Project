from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from recipe.single_turn_reward.v3.reward_model_stage3_ab_cloud import compare_output_vs_gts


_GENRM_TRACE_LOCK = Lock()


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _append_genrm_trace(path: str, record: dict[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _GENRM_TRACE_LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


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


def _extract_gts(extra_info: dict[str, Any] | None, ground_truth: str) -> str:
    extra_info = extra_info or {}
    # Prefer explicit gts fields from dataset.
    for key in ("gts", "gt", "reference_answer", "target", "gold_answer"):
        v = extra_info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fallback to current ground_truth if caller stores reference response there.
    return (ground_truth or "").strip()


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
    output_answer = solution_str or ""
    gts_answer = _extract_gts(extra_info, ground_truth)
    collect_genrm_io = _to_bool(kwargs.get("collect_genrm_io", False), default=False)
    genrm_io_path = str(kwargs.get("genrm_io_path", "/tmp/genrm_io.jsonl"))
    include_extra_info = _to_bool(kwargs.get("genrm_io_include_extra_info", False), default=False)

    enable_model_judge = bool(kwargs.get("enable_model_judge", True))
    if enable_model_judge:
        model_res = await compare_output_vs_gts(
            question=q,
            output_answer=output_answer,
            gts_answer=gts_answer,
            api_base=str(kwargs.get("api_base", "")),
            api_key=str(kwargs.get("api_key", "")),
            judge_model=str(kwargs.get("judge_model", "qwen-max")),
            timeout_s=float(kwargs.get("model_judge_timeout_s", 45.0)),
            history_text=_history_text(extra_info),
            ab_max_retries=int(kwargs.get("ab_max_retries", 5)),
            ab_backoff_base_s=float(kwargs.get("ab_backoff_base_s", 0.8)),
            ab_backoff_max_s=float(kwargs.get("ab_backoff_max_s", 15.0)),
        )
    else:
        model_res = {
            "score": 0.5,
            "winner": "tie",
            "score_output": 0.5,
            "score_gts": 0.5,
            "status": "disabled",
            "reason": "",
            "raw": "",
        }

    reward_score = _clip(float(model_res.get("score", 0.5)))

    if collect_genrm_io:
        trace_record: dict[str, Any] = {
            "event": "genrm_io",
            "question": q,
            "output": output_answer,
            "gts": gts_answer,
            "ab_winner": str(model_res.get("winner", "tie")),
            "ab_score_output": _clip(float(model_res.get("score_output", 0.5))),
            "ab_score_gts": _clip(float(model_res.get("score_gts", 0.5))),
            "score": reward_score,
            "model_judge_status": str(model_res.get("status", "")),
            "ab_reason": str(model_res.get("reason", ""))[:300],
        }
        if include_extra_info:
            trace_record["extra_info"] = extra_info
        _append_genrm_trace(genrm_io_path, trace_record)

    return {
        "score": reward_score,
        "rule_score": 0.0,
        "model_judge_score": reward_score,
        "model_judge_status": str(model_res.get("status", "")),
        "ab_winner": str(model_res.get("winner", "tie")),
        "ab_score_output": _clip(float(model_res.get("score_output", 0.5))),
        "ab_score_gts": _clip(float(model_res.get("score_gts", 0.5))),
        "ab_reason": str(model_res.get("reason", ""))[:120],
        "rule_weight": 0.0,
        "model_weight": 1.0,
    }
