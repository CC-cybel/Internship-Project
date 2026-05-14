from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

from recipe.single_turn_reward.v5.reward_model_stage5_mid_cloud import score_output_mid_stage


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


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
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


def _extract_final_block(text: str) -> str:
    """Extract content between BEGIN_FINAL and END_FINAL."""
    m = re.search(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL", text or "")
    return m.group(1).strip() if m else text.strip()


def _system_prompt(extra_info: dict[str, Any] | None) -> str:
    extra_info = extra_info or {}
    for key in ("transformed_system_prompt", "original_system_prompt", "system_prompt"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _history_text(extra_info: dict[str, Any] | None) -> str:
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
            role, text = item
            # Match v4: assistant history only exposes user-visible final text.
            if role == "assistant":
                text = _extract_final_block(text)
                if not text:
                    continue
            normalized.append((role, text))
    if not normalized:
        return ""

    # Match v4: keep the complete dialogue history available in extra_info.
    lines = [f"{r}: {c}" for r, c in normalized]
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
    output_answer = solution_str or ""
    collect_genrm_io = _to_bool(kwargs.get("collect_genrm_io", False), default=False)
    genrm_io_path = str(kwargs.get("genrm_io_path", "/tmp/genrm_io.jsonl"))
    include_extra_info = _to_bool(kwargs.get("genrm_io_include_extra_info", False), default=False)

    # 从 extra_info 中获取 turn_round 和 contact_round
    ei = extra_info or {}
    turn_round = int(_to_float(ei.get("turn_round"), 0))
    contact_round = int(_to_float(ei.get("rule_contact_round", ei.get("contact_round")), 0))
    system_prompt = _system_prompt(extra_info)

    enable_model_judge = bool(kwargs.get("enable_model_judge", True))
    if enable_model_judge:
        model_res = await score_output_mid_stage(
            question=q,
            output_answer=output_answer,
            api_base=str(kwargs.get("api_base", "")),
            api_key=str(kwargs.get("api_key", "")),
            judge_model=str(kwargs.get("judge_model", "qwen-max")),
            timeout_s=float(kwargs.get("model_judge_timeout_s", 45.0)),
            history_text=_history_text(extra_info),
            system_prompt=system_prompt,
            turn_round=turn_round,
            contact_round=contact_round,
            score_max_retries=int(kwargs.get("score_max_retries", kwargs.get("ab_max_retries", 5))),
            score_backoff_base_s=float(kwargs.get("score_backoff_base_s", kwargs.get("ab_backoff_base_s", 0.8))),
            score_backoff_max_s=float(kwargs.get("score_backoff_max_s", kwargs.get("ab_backoff_max_s", 15.0))),
        )
    else:
        model_res = {
            "score": 0.5,
            "status": "disabled",
            "reason": "",
            "raw": "",
        }

    reward_score = _clip(float(model_res.get("score", 0.5)))
    model_judge_score_raw = _clip(_to_float(model_res.get("model_judge_score_raw"), reward_score))
    length_penalty = _clip(_to_float(model_res.get("length_penalty"), 0.0))
    question_penalty = _clip(_to_float(model_res.get("question_penalty"), 0.0))
    sep_penalty = _clip(_to_float(model_res.get("sep_penalty"), 0.0))
    filler_word_penalty = _clip(_to_float(model_res.get("filler_word_penalty"), 0.0))
    diagnosis_penalty = _clip(_to_float(model_res.get("diagnosis_penalty"), 0.0))
    filler_word_hits = model_res.get("filler_word_hits", [])
    diagnosis_penalty_rules = model_res.get("diagnosis_penalty_rules", [])
    filler_word_hits_json = json.dumps(filler_word_hits, ensure_ascii=False)
    diagnosis_penalty_rules_json = json.dumps(diagnosis_penalty_rules, ensure_ascii=False)

    if collect_genrm_io:
        trace_record: dict[str, Any] = {
            "event": "genrm_io",
            "question": q,
            "output": output_answer,
            "score": reward_score,
            "model_judge_score_raw": model_judge_score_raw,
            "length_penalty": length_penalty,
            "question_penalty": question_penalty,
            "sep_penalty": sep_penalty,
            "filler_word_penalty": filler_word_penalty,
            "filler_word_hits": filler_word_hits,
            "diagnosis_penalty": diagnosis_penalty,
            "diagnosis_penalty_rules": diagnosis_penalty_rules,
            "diagnosis_penalty_reason": str(model_res.get("diagnosis_penalty_reason", ""))[:120],
            "model_judge_status": str(model_res.get("status", "")),
            "single_score_reason": str(model_res.get("reason", ""))[:300],
        }
        if include_extra_info:
            trace_record["extra_info"] = extra_info
        _append_genrm_trace(genrm_io_path, trace_record)

    return {
        "score": reward_score,
        "rule_score": 0.0,
        "model_judge_score": reward_score,
        "model_judge_score_raw": model_judge_score_raw,
        "length_penalty": length_penalty,
        "question_penalty": question_penalty,
        "sep_penalty": sep_penalty,
        "filler_word_penalty": filler_word_penalty,
        "filler_word_hits": filler_word_hits_json,
        "diagnosis_penalty": diagnosis_penalty,
        "diagnosis_penalty_rules": diagnosis_penalty_rules_json,
        "diagnosis_penalty_reason": str(model_res.get("diagnosis_penalty_reason", ""))[:120],
        "model_judge_status": str(model_res.get("status", "")),
        "ab_winner": "single_score_mode",
        "ab_score_output": reward_score,
        "ab_score_gts": 0.0,
        "ab_reason": str(model_res.get("reason", ""))[:120],
        "rule_weight": 0.0,
        "model_weight": 1.0,
    }
