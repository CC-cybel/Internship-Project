"""Single-turn hybrid reward template for verl Reward Loop.

This template combines:
1) Generative judge for lead reward
2) Generative judge for instruction following
3) Rule-based format reward (regex/json/prefix/suffix checks)

Two judge prompt templates are intentionally left empty. Fill them before training.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import re
import time
from threading import Lock
from typing import Any

import aiohttp

# Fill these two templates later.
# Available placeholders:
# - {question}
# - {answer}
# - {ground_truth}
# - {extra_info_json}
LEAD_JUDGE_PROMPT_TEMPLATE = ""
INSTRUCTION_JUDGE_PROMPT_TEMPLATE = ""
DEFAULT_LEAD_REWARD_FUNC_PATH = os.path.join(os.path.dirname(__file__), "lead_reward.py")
DEFAULT_INSTRUCTION_REWARD_FUNC_PATH = os.path.join(os.path.dirname(__file__), "instruction_reward.py")

_EXTERNAL_FN_CACHE: dict[tuple[str, str], Any] = {}
_GENRM_TRACE_LOCK = Lock()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _append_genrm_trace(path: str, record: dict[str, Any]) -> None:
    if not path:
        return
    try:
        trace_path = os.path.abspath(os.path.expanduser(path))
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with _GENRM_TRACE_LOCK:
            fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
    except Exception:
        # Do not block training on trace write failures.
        pass


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _extract_first_float(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _parse_judge_score(raw_text: str, score_max: float) -> float:
    """Parse judge output and normalize to [0, 1]."""
    raw_text = (raw_text or "").strip()
    parsed: float | None = None

    # Try JSON first.
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and "score" in data:
            parsed = _to_float(data["score"], default=0.0)
        elif isinstance(data, (int, float)):
            parsed = float(data)
    except Exception:
        pass

    # Fallback to regex number extraction.
    if parsed is None:
        parsed = _extract_first_float(raw_text)

    if parsed is None:
        return 0.0
    if score_max <= 0:
        return _clip(parsed)
    return _clip(parsed / score_max)


def _extract_question(extra_info: dict[str, Any], ground_truth: str) -> str:
    for key in ("question", "instruction", "prompt", "query"):
        val = extra_info.get(key) if isinstance(extra_info, dict) else None
        if isinstance(val, str) and val.strip():
            return val
    # Fallback when question is not provided in dataset.
    return ground_truth if isinstance(ground_truth, str) else ""


def _compute_format_score(solution_str: str, extra_info: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Compute format score with fixed hard_dual_s5_full rules only."""
    del extra_info
    del kwargs
    return _compute_hard_dual_s5_full_format_score(solution_str)


def _compute_hard_dual_s5_full_format_score(solution_str: str) -> dict[str, Any]:
    """Format profile derived from LlamaFactory/data/hard_dual_s5_full.json.

    Required structure:
    BEGIN_META
    action=...
    thought=...
    ... (optional key=value lines, usually slot_*)
    END_META
    BEGIN_FINAL
    ...
    END_FINAL
    """
    text = (solution_str or "").strip()
    checks: dict[str, bool] = {}

    checks["single_markers"] = (
        text.count("BEGIN_META") == 1
        and text.count("END_META") == 1
        and text.count("BEGIN_FINAL") == 1
        and text.count("END_FINAL") == 1
    )

    pattern = re.compile(
        r"^BEGIN_META\n(?P<meta>[\s\S]*?)\nEND_META\nBEGIN_FINAL\n(?P<final>[\s\S]*?)\nEND_FINAL$",
        flags=re.DOTALL,
    )
    matched = pattern.match(text)
    checks["block_order"] = matched is not None

    meta = ""
    final = ""
    if matched is not None:
        meta = matched.group("meta")
        final = matched.group("final")

    meta_lines = [line for line in meta.split("\n") if line != ""]
    checks["meta_non_empty"] = len(meta_lines) >= 2
    checks["action_first"] = len(meta_lines) >= 1 and meta_lines[0].startswith("action=")
    checks["thought_second"] = len(meta_lines) >= 2 and meta_lines[1].startswith("thought=")
    checks["all_meta_key_value"] = all("=" in line for line in meta_lines)
    checks["meta_key_ascii"] = all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", line) is not None for line in meta_lines)
    checks["final_non_empty"] = final.strip() != ""

    score = sum(1.0 if ok else 0.0 for ok in checks.values()) / max(len(checks), 1)
    detail = {
        **checks,
        "meta_line_count": len(meta_lines),
        "has_slot_line": any(line.startswith("slot_") for line in meta_lines),
    }
    return {"score": _clip(score), "detail": detail}


def _load_external_callable(module_path: str, function_name: str) -> Any:
    key = (module_path, function_name)
    if key in _EXTERNAL_FN_CACHE:
        return _EXTERNAL_FN_CACHE[key]

    normalized = module_path
    if normalized.startswith("file://"):
        normalized = normalized[len("file://") :]
    normalized = os.path.expanduser(normalized)
    if not os.path.isabs(normalized):
        normalized = os.path.abspath(normalized)

    if not os.path.exists(normalized):
        raise FileNotFoundError(f"external reward module not found: {normalized}")

    module_name = f"single_turn_reward_ext_{abs(hash(normalized))}"
    spec = importlib.util.spec_from_file_location(module_name, normalized)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from: {normalized}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"{function_name} not found in {normalized}")
    fn = getattr(module, function_name)
    _EXTERNAL_FN_CACHE[key] = fn
    return fn


async def _run_external_evaluator(
    *,
    module_path: str,
    function_name: str,
    question: str,
    answer: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        fn = _load_external_callable(module_path=module_path, function_name=function_name)
        call_kwargs = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "reward_router_address": reward_router_address,
            **kwargs,
        }
        if inspect.iscoroutinefunction(fn):
            result = await fn(**call_kwargs)
        else:
            result = await asyncio.to_thread(fn, **call_kwargs)

        if isinstance(result, dict):
            return {
                "score": _clip(_to_float(result.get("score", 0.0), default=0.0)),
                "raw": str(result.get("raw", "")),
                "status": str(result.get("status", "ok_external")),
            }
        return {
            "score": _clip(_to_float(result, default=0.0)),
            "raw": "",
            "status": "ok_external",
        }
    except Exception as exc:
        return {"score": 0.0, "raw": "", "status": f"external_error:{type(exc).__name__}"}


async def _chat_complete(
    router_address: str,
    request: dict[str, Any],
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    base = str(router_address).strip()
    if base.startswith("http://") or base.startswith("https://"):
        base_url = base.rstrip("/")
    else:
        base_url = f"http://{base.rstrip('/')}"
    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=request, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_backoff_s * (2**attempt))
    assert last_exc is not None
    raise last_exc


async def _run_judge(
    *,
    template: str,
    question: str,
    answer: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None,
    model_name: str | None,
    score_max: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    if not template.strip():
        return {"score": 0.0, "raw": "", "status": "prompt_template_empty"}
    if not reward_router_address:
        return {"score": 0.0, "raw": "", "status": "reward_router_address_missing"}
    if not model_name:
        return {"score": 0.0, "raw": "", "status": "judge_model_missing"}

    prompt = template.format(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        extra_info_json=json.dumps(extra_info, ensure_ascii=False),
    )
    request = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    try:
        response = await _chat_complete(
            router_address=reward_router_address,
            request=request,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            api_key=api_key,
        )
        raw = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return {"score": _parse_judge_score(raw, score_max=score_max), "raw": raw, "status": "ok"}
    except Exception as exc:
        return {"score": 0.0, "raw": "", "status": f"judge_error:{type(exc).__name__}"}


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main reward entry for verl Reward Loop.

    Return format must contain key `score`.
    """
    del data_source
    del reward_model_tokenizer

    extra_info = extra_info or {}
    question = _extract_question(extra_info, ground_truth)
    answer = solution_str or ""

    format_result = _compute_format_score(answer, extra_info=extra_info, kwargs=kwargs)
    hard_format_gate = bool(kwargs.get("hard_format_gate", True))
    format_gate_threshold = _to_float(kwargs.get("format_gate_threshold", 1.0), default=1.0)
    format_fail_score = _clip(_to_float(kwargs.get("format_fail_score", 0.0), default=0.0))
    collect_genrm_io = _to_bool(kwargs.get("collect_genrm_io", False), default=False)
    genrm_io_path = str(kwargs.get("genrm_io_path", "/tmp/genrm_io.jsonl"))
    include_extra_info = _to_bool(kwargs.get("genrm_io_include_extra_info", False), default=False)
    sample_id = str(extra_info.get("sample_id", "")) if isinstance(extra_info, dict) else ""

    if hard_format_gate and format_result["score"] < format_gate_threshold:
        gated_result = {
            "score": format_fail_score,
            "lead_score": 0.0,
            "instruction_follow_score": 0.0,
            "format_score": _clip(format_result["score"]),
            "lead_status": "skipped_due_to_format_gate",
            "instruction_status": "skipped_due_to_format_gate",
            "lead_judge_raw": "",
            "instruction_judge_raw": "",
            "format_detail": format_result["detail"],
            "weight_lead": 0.0,
            "weight_instruction": 0.0,
            "weight_format": 1.0,
            "hard_format_gate": True,
            "gated": True,
        }
        if collect_genrm_io:
            _append_genrm_trace(
                genrm_io_path,
                {
                    "ts": time.time(),
                    "event": "genrm_summary",
                    "sample_id": sample_id,
                    "gated": True,
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "format_score": gated_result["format_score"],
                    "format_detail": gated_result["format_detail"],
                    "result": gated_result,
                    "extra_info": extra_info if include_extra_info else {"sample_id": sample_id},
                },
            )
        return gated_result

    # Weights (normalized by default).
    lead_weight = _to_float(kwargs.get("lead_weight", 0.4), default=0.4)
    instruction_weight = _to_float(kwargs.get("instruction_weight", 0.4), default=0.4)
    format_weight = _to_float(kwargs.get("format_weight", 0.2), default=0.2)
    normalize_weights = bool(kwargs.get("normalize_weights", True))

    # Judge model + sampling config.
    default_model = kwargs.get("judge_model")
    lead_model = kwargs.get("lead_judge_model", default_model)
    instruction_model = kwargs.get("instruction_judge_model", default_model)

    lead_score_max = _to_float(kwargs.get("lead_judge_score_max", 10.0), default=10.0)
    instruction_score_max = _to_float(kwargs.get("instruction_judge_score_max", 10.0), default=10.0)

    timeout_s = _to_float(kwargs.get("judge_timeout_s", 240.0), default=240.0)
    max_retries = int(kwargs.get("judge_max_retries", 3))
    retry_backoff_s = _to_float(kwargs.get("judge_retry_backoff_s", 1.0), default=1.0)
    api_key = kwargs.get("api_key") or kwargs.get("llm_api_key")

    lead_template = str(kwargs.get("lead_judge_prompt_template", LEAD_JUDGE_PROMPT_TEMPLATE))
    instruction_template = str(
        kwargs.get("instruction_judge_prompt_template", INSTRUCTION_JUDGE_PROMPT_TEMPLATE)
    )
    lead_func_path = kwargs.get("lead_reward_func_path", DEFAULT_LEAD_REWARD_FUNC_PATH)
    lead_func_name = str(kwargs.get("lead_reward_func_name", "compute_lead_score"))
    if isinstance(lead_func_path, str) and lead_func_path.strip():
        lead_task = _run_external_evaluator(
            module_path=str(lead_func_path),
            function_name=lead_func_name,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_router_address=reward_router_address,
            kwargs=kwargs,
        )
    else:
        lead_task = _run_judge(
            template=lead_template,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_router_address=reward_router_address,
            model_name=lead_model,
            score_max=lead_score_max,
            temperature=_to_float(kwargs.get("lead_judge_temperature", 0.0), default=0.0),
            top_p=_to_float(kwargs.get("lead_judge_top_p", 1.0), default=1.0),
            max_tokens=int(kwargs.get("lead_judge_max_tokens", 64)),
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            api_key=api_key,
        )

    instruction_func_path = kwargs.get("instruction_reward_func_path", DEFAULT_INSTRUCTION_REWARD_FUNC_PATH)
    instruction_func_name = str(kwargs.get("instruction_reward_func_name", "compute_instruction_score"))
    if instruction_func_path:
        instruction_task = _run_external_evaluator(
            module_path=str(instruction_func_path),
            function_name=instruction_func_name,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_router_address=reward_router_address,
            kwargs=kwargs,
        )
    else:
        instruction_task = _run_judge(
            template=instruction_template,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_router_address=reward_router_address,
            model_name=instruction_model,
            score_max=instruction_score_max,
            temperature=_to_float(kwargs.get("instruction_judge_temperature", 0.0), default=0.0),
            top_p=_to_float(kwargs.get("instruction_judge_top_p", 1.0), default=1.0),
            max_tokens=int(kwargs.get("instruction_judge_max_tokens", 64)),
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            api_key=api_key,
        )

    lead_result, instruction_result = await asyncio.gather(lead_task, instruction_task)

    if normalize_weights:
        weight_sum = lead_weight + instruction_weight + format_weight
        if weight_sum > 0:
            lead_weight /= weight_sum
            instruction_weight /= weight_sum
            format_weight /= weight_sum

    total_score = (
        lead_weight * lead_result["score"]
        + instruction_weight * instruction_result["score"]
        + format_weight * format_result["score"]
    )

    result = {
        "score": _clip(total_score),
        "lead_score": _clip(lead_result["score"]),
        "instruction_follow_score": _clip(instruction_result["score"]),
        "format_score": _clip(format_result["score"]),
        "lead_status": lead_result["status"],
        "instruction_status": instruction_result["status"],
        "lead_judge_raw": lead_result["raw"],
        "instruction_judge_raw": instruction_result["raw"],
        "format_detail": format_result["detail"],
        "weight_lead": lead_weight,
        "weight_instruction": instruction_weight,
        "weight_format": format_weight,
        "hard_format_gate": hard_format_gate,
        "gated": False,
    }
    if collect_genrm_io:
        _append_genrm_trace(
            genrm_io_path,
            {
                "ts": time.time(),
                "event": "genrm_summary",
                "sample_id": sample_id,
                "gated": False,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "lead_result": lead_result,
                "instruction_result": instruction_result,
                "format_result": format_result,
                "result": result,
                "extra_info": extra_info if include_extra_info else {"sample_id": sample_id},
            },
        )
    return result
