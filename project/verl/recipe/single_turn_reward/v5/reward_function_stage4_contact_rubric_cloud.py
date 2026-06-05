from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from recipe.single_turn_reward.v5.reward_model_stage4_contact_rubric_cloud import score_output_contact_rubric


_GENRM_TRACE_LOCK = Lock()
_SFT_TRACE_LOCK = Lock()


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


def _append_jsonl(path: str, record: dict[str, Any], lock: Lock) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with lock:
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
    import re

    m = re.search(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL", text or "")
    return m.group(1).strip() if m else text.strip()


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
        if item is None:
            continue
        role, text = item
        if role == "assistant":
            text = _extract_final_block(text)
            if not text:
                continue
        normalized.append((role, text))
    return "\n".join(f"{r}: {c}" for r, c in normalized)


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
    extra = extra_info or {}
    system_prompt = str(extra.get("transformed_system_prompt", "")).strip()
    history_text = _history_text(extra_info)

    enable_model_judge = _to_bool(kwargs.get("enable_model_judge", True), default=True)
    if enable_model_judge:
        model_res = await score_output_contact_rubric(
            question=q,
            output_answer=output_answer,
            api_base=str(kwargs.get("api_base", "")),
            api_key=str(kwargs.get("api_key", "")),
            judge_model=str(kwargs.get("judge_model", "qwen-max")),
            timeout_s=float(kwargs.get("model_judge_timeout_s", 45.0)),
            history_text=history_text,
            system_prompt=system_prompt,
            rubric_path=kwargs.get("rubric_path"),
            rubric_index_path=kwargs.get("rubric_index_path"),
            hard_config_path=kwargs.get("hard_config_path"),
            judge_max_tokens=int(kwargs.get("judge_max_tokens", 1600)),
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
            "rubric_results": [],
        }

    reward_score = _clip(float(model_res.get("score", 0.5)))
    semantic_score = _clip(float(model_res.get("semantic_score", model_res.get("model_judge_score_raw", reward_score))))

    collect_genrm_io = _to_bool(kwargs.get("collect_genrm_io", False), default=False)
    if collect_genrm_io:
        trace_record: dict[str, Any] = {
            "event": "genrm_io",
            "question": q,
            "output": output_answer,
            "score": reward_score,
            "semantic_score": semantic_score,
            "model_judge_score_raw": model_res.get("model_judge_score_raw"),
            "rubric_version": model_res.get("rubric_version"),
            "rubric_path": model_res.get("rubric_path"),
            "rubric_results": model_res.get("rubric_results", []),
            "gate_failed": model_res.get("gate_failed", False),
            "gate_reason": str(model_res.get("gate_reason", ""))[:160],
            "model_judge_status": str(model_res.get("status", "")),
            "single_score_reason": str(model_res.get("reason", ""))[:300],
            "final_char_len": model_res.get("final_char_len"),
            "length_penalty": model_res.get("length_penalty"),
            "sep_penalty": model_res.get("sep_penalty"),
            "banned_term_penalty": model_res.get("banned_term_penalty"),
            "banned_term_hits": model_res.get("banned_term_hits"),
            "hard_penalty_total": model_res.get("hard_penalty_total"),
        }
        if _to_bool(kwargs.get("genrm_io_include_extra_info", False), default=False):
            trace_record["extra_info"] = extra_info
        _append_jsonl(str(kwargs.get("genrm_io_path", "/tmp/genrm_io.jsonl")), trace_record, _GENRM_TRACE_LOCK)

    if _to_bool(kwargs.get("save_high_score_sft", False), default=False):
        threshold = float(kwargs.get("sft_score_threshold", 0.9))
        if reward_score >= threshold:
            history_messages = []
            if system_prompt:
                history_messages.append({"role": "system", "content": system_prompt})
            for turn in extra.get("conversations", [])[:-1]:
                item = _normalize_turn(turn)
                if item is not None:
                    role, text = item
                    history_messages.append({"role": role, "content": text})
            sft_record = {"messages": history_messages + [{"role": "assistant", "content": output_answer}]}
            _append_jsonl(str(kwargs.get("sft_output_path", "/tmp/high_score_sft.jsonl")), sft_record, _SFT_TRACE_LOCK)

    return {
        "score": reward_score,
        "rule_score": 1.0 if not model_res.get("gate_failed") else 0.0,
        "model_judge_score": reward_score,
        "model_judge_score_raw": model_res.get("model_judge_score_raw", semantic_score),
        "semantic_score": semantic_score,
        "model_judge_status": str(model_res.get("status", "")),
        "ab_winner": "rubric_score_mode",
        "ab_score_output": reward_score,
        "ab_score_gts": 0.0,
        "ab_reason": str(model_res.get("reason", ""))[:120],
        "rubric_version": str(model_res.get("rubric_version", "")),
        "gate_failed": 1.0 if model_res.get("gate_failed") else 0.0,
        "gate_reason": str(model_res.get("gate_reason", ""))[:120],
        "final_char_len": model_res.get("final_char_len", 0),
        "length_penalty": model_res.get("length_penalty", 0.0),
        "sep_penalty": model_res.get("sep_penalty", 0.0),
        "banned_term_penalty": model_res.get("banned_term_penalty", 0.0),
        "banned_term_hits": json.dumps(model_res.get("banned_term_hits", []), ensure_ascii=False),
        "hard_penalty_total": model_res.get("hard_penalty_total", 0.0),
        "rubric_results": json.dumps(model_res.get("rubric_results", []), ensure_ascii=False),
        "rule_weight": 0.0,
        "model_weight": 1.0,
    }
