from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from recipe.single_turn_reward.v2.reward_model_stage4_contact_cloud import score_output_contact_only


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


def _append_genrm_trace(path: str, record: dict[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _GENRM_TRACE_LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _append_sft_trace(path: str, record: dict[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _SFT_TRACE_LOCK:
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
        if item is not None:
            role, text = item
            # assistant 回复只保留 BEGIN_FINAL~END_FINAL 片段
            if role == "assistant":
                text = _extract_final_block(text)
                if not text:
                    continue
            normalized.append((role, text))
    if not normalized:
        return ""

    # 读取所有轮次
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

    # 高分样本保存配置
    save_high_score_sft = _to_bool(kwargs.get("save_high_score_sft", False), default=False)
    sft_score_threshold = float(kwargs.get("sft_score_threshold", 0.9))
    sft_output_path = str(kwargs.get("sft_output_path", "/tmp/high_score_sft.jsonl"))

    extra = extra_info or {}
    system_prompt = str(extra.get("transformed_system_prompt", "")).strip()
    history_text = _history_text(extra_info)

    enable_model_judge = _to_bool(kwargs.get("enable_model_judge", True), default=True)
    if enable_model_judge:
        model_res = await score_output_contact_only(
            question=q,
            output_answer=output_answer,
            api_base=str(kwargs.get("api_base", "")),
            api_key=str(kwargs.get("api_key", "")),
            judge_model=str(kwargs.get("judge_model", "qwen-max")),
            timeout_s=float(kwargs.get("model_judge_timeout_s", 45.0)),
            history_text=history_text,
            system_prompt=system_prompt,
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

    judge_score = _clip(float(model_res.get("score", 0.5)))
    judge_score_raw = _clip(float(model_res.get("model_judge_score_raw", judge_score)))
    age_contact_pass = 1.0 if _to_bool(model_res.get("age_contact_pass", 1), default=True) else 0.0
    reward_score = _clip(judge_score * age_contact_pass)

    if collect_genrm_io:
        trace_record: dict[str, Any] = {
            "event": "genrm_io",
            "question": q,
            "output": output_answer,
            "score": reward_score,
            "model_judge_score_raw": judge_score_raw,
            "model_judge_score_after_penalty": judge_score,
            "age_contact_pass": int(age_contact_pass),
            "age_contact_reason": str(model_res.get("age_contact_reason", ""))[:120],
            "model_judge_status": str(model_res.get("status", "")),
            "single_score_reason": str(model_res.get("reason", ""))[:300],
            "final_char_len": model_res.get("final_char_len"),
            "length_penalty": model_res.get("length_penalty"),
            "sep_penalty": model_res.get("sep_penalty"),
            "filler_word_penalty": model_res.get("filler_word_penalty"),
            "filler_word_hits": model_res.get("filler_word_hits"),
        }
        if include_extra_info:
            trace_record["extra_info"] = extra_info
        _append_genrm_trace(genrm_io_path, trace_record)

    # 保存高分样本到SFT格式
    if save_high_score_sft and reward_score >= sft_score_threshold:
        # 构建history: extra_info.conversations中除了最后一条assistant回复外的所有对话
        conversations = extra.get("conversations", [])

        history_messages = []
        if system_prompt:
            history_messages.append({"role": "system", "content": system_prompt})

        # conversations的最后一条是当前要预测的assistant回复，不加入history
        for turn in conversations[:-1]:
            item = _normalize_turn(turn)
            if item is not None:
                role, text = item
                history_messages.append({"role": role, "content": text})

        sft_record = {
            "messages": history_messages + [{"role": "assistant", "content": output_answer}]
        }
        _append_sft_trace(sft_output_path, sft_record)

    return {
        "score": reward_score,
        "rule_score": age_contact_pass,
        "model_judge_score": judge_score,
        "model_judge_score_raw": judge_score_raw,
        "model_judge_status": str(model_res.get("status", "")),
        "ab_winner": "single_score_mode",
        "ab_score_output": reward_score,
        "ab_score_gts": 0.0,
        "ab_reason": str(model_res.get("reason", ""))[:120],
        "age_contact_pass": age_contact_pass,
        "age_contact_reason": str(model_res.get("age_contact_reason", ""))[:120],
        "final_char_len": model_res.get("final_char_len", 0),
        "length_penalty": model_res.get("length_penalty", 0.0),
        "sep_penalty": model_res.get("sep_penalty", 0.0),
        "filler_word_penalty": model_res.get("filler_word_penalty", 0.0),
        "filler_word_hits": json.dumps(model_res.get("filler_word_hits", []), ensure_ascii=False),
        "rule_weight": 0.0,
        "model_weight": 1.0,
    }
