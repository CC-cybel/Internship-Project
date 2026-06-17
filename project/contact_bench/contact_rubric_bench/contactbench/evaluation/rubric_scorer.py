from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import aiohttp


_FINAL_BLOCK_RE = re.compile(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL")
_META_BLOCK_RE = re.compile(r"BEGIN_META\s*\n([\s\S]*?)\nEND_META")
_HERE = Path(__file__).resolve().parent
_BENCH_ROOT = _HERE.parents[1]
_DEFAULT_RUBRIC_INDEX = _BENCH_ROOT / "data" / "rubrics" / "rubric_index.json"
_DEFAULT_HARD_CONFIG = _BENCH_ROOT / "data" / "rules" / "contact_reward_hard_config.json"


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    l = text.find("{")
    r = text.rfind("}")
    if l < 0 or r < 0 or r <= l:
        return None
    try:
        obj = json.loads(text[l : r + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_final_text(answer: str) -> str:
    m = _FINAL_BLOCK_RE.search(answer or "")
    if not m:
        return ""
    return m.group(1).strip()


def _extract_meta_text(answer: str) -> str:
    m = _META_BLOCK_RE.search(answer or "")
    if not m:
        return ""
    return m.group(1).strip()


def _active_rubric_path(index_path: str | Path | None = None) -> Path:
    index_file = Path(index_path).expanduser() if index_path else _DEFAULT_RUBRIC_INDEX
    index = _load_json(index_file)
    active = str(index.get("active_version", "")).strip()
    if not active:
        raise ValueError(f"Missing active_version in {index_file}")
    return (index_file.parent / active).resolve()


def load_active_rubric_set(
    rubric_path: str | Path | None = None,
    rubric_index_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    path = Path(rubric_path).expanduser().resolve() if rubric_path else _active_rubric_path(rubric_index_path)
    data = _load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("rubrics"), list):
        raise ValueError(f"Invalid rubric set: {path}")
    return data, path


def load_hard_config(path: str | Path | None = None) -> dict[str, Any]:
    data = _load_json(path or _DEFAULT_HARD_CONFIG)
    if not isinstance(data, dict):
        raise ValueError("Hard config must be a JSON object")
    return data


def _length_penalty(char_len: int, config: dict[str, Any]) -> float:
    cfg = config.get("length", {}) if isinstance(config, dict) else {}
    lower = int(cfg.get("lower", 90))
    upper = int(cfg.get("upper", 160))
    too_short = int(cfg.get("too_short", 60))
    too_long = int(cfg.get("too_long", 200))
    max_penalty = float(cfg.get("max_penalty", 0.10))

    if char_len <= 0:
        return max_penalty
    if lower <= char_len <= upper:
        return 0.0
    if char_len < lower:
        span = max(1, lower - too_short)
        return min(max_penalty, (max(0, lower - char_len) / span) * max_penalty)
    span = max(1, too_long - upper)
    return min(max_penalty, (max(0, char_len - upper) / span) * max_penalty)


def _sep_penalty(answer: str, config: dict[str, Any]) -> float:
    cfg = config.get("sep", {}) if isinstance(config, dict) else {}
    missing_penalty = float(cfg.get("missing_penalty", 0.01))
    final_text = _extract_final_text(answer)
    text = final_text if final_text else (answer or "")
    return 0.0 if "<sep>" in text else missing_penalty


def _banned_term_penalty(text: str, config: dict[str, Any]) -> tuple[float, list[str]]:
    cfg = config.get("banned_terms", {}) if isinstance(config, dict) else {}
    terms = cfg.get("terms", [])
    if not isinstance(terms, list):
        terms = []
    penalty = float(cfg.get("penalty", 0.10))
    match_type = str(cfg.get("match_type", "literal")).strip().lower()

    if match_type != "literal":
        # Keep this deterministic. New match modes should be explicit code, not LLM behavior.
        match_type = "literal"

    hits = [str(term) for term in terms if str(term) and str(term) in (text or "")]
    if not hits:
        return 0.0, []
    return penalty, hits


def compute_hard_penalties(output_answer: str, hard_config: dict[str, Any]) -> dict[str, Any]:
    final_text = _extract_final_text(output_answer)
    final_char_len = len(final_text)
    len_pen = _length_penalty(final_char_len, hard_config)
    sep_pen = _sep_penalty(output_answer, hard_config)
    banned_pen, banned_hits = _banned_term_penalty(final_text, hard_config)
    total = len_pen + sep_pen + banned_pen
    return {
        "final_char_len": final_char_len,
        "length_penalty": len_pen,
        "sep_penalty": sep_pen,
        "banned_term_penalty": banned_pen,
        "banned_term_hits": banned_hits,
        "hard_penalty_total": total,
    }


async def _chat(api_base: str, api_key: str, payload: dict[str, Any], timeout_s: float = 45.0) -> dict[str, Any]:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=text[:300],
                    headers=resp.headers,
                )
            return json.loads(text)


def _compact_rubrics(rubric_set: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in rubric_set.get("rubrics", []):
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        compact.append(
            {
                "id": item.get("id"),
                "name_cn": item.get("name_cn"),
                "stage": item.get("stage"),
                "rule_type": item.get("rule_type"),
                "weight": item.get("weight"),
                "trigger_condition": item.get("trigger_condition"),
                "description": item.get("description"),
                "score_levels": item.get("score_levels"),
                "fail_policy": item.get("fail_policy"),
            }
        )
    return compact


def _build_rubric_prompt(
    *,
    question: str,
    answer_final_block: str,
    answer_meta_block: str,
    history_block: str,
    system_prompt: str,
    rubric_set: dict[str, Any],
) -> str:
    rubrics = _compact_rubrics(rubric_set)
    rubric_ids = [str(r.get("id")) for r in rubrics]
    system_block = f"系统提示:\n{system_prompt}\n\n" if system_prompt else ""
    return (
        "你是严格的医疗咨询转化评审员，专门评估留联阶段当前回复。\n"
        "评分对象只能是【当前候选回复_FINAL】。历史轮次只用于判断是否首次/再次留联、用户信息、拒绝与承接；【当前候选回复_META】只用于检查策略、槽位、用户模型与 FINAL 是否一致，不能把 META 当成用户可见话术加分。\n"
        "请逐条检查 Rubrics。不要把硬性机械惩罚写入评分，这些由程序另算。\n\n"
        "评分原则:\n"
        "1) 每条 rubric 必须返回一条结果，rubric_id 必须完整覆盖 active_rubric_ids。\n"
        "2) triggered=false 表示该 rubric 当前不适用；triggered=true 时 score 只能是 0、1、2。\n"
        "3) 对 gate 类 rubric，如果明确违反，score=0 且 gate_failed=true。\n"
        "4) 如果当前回复只是“留个电话/微信做评估/给方案”，缺少针对用户具体情况的福利，不要给高分。\n"
        "5) 索要联系方式时又继续问新的症状、病史、睡眠、用药等问题，应在相关 rubric 给低分。\n"
        "6) strategy_slot_consistency 必须结合 META 中的 thought、slot、策略描述和 FINAL 进行判断；如果 META 与历史矛盾或 FINAL 不一致，应触发并低分。\n"
        "7) 输出必须是 JSON，禁止额外文本。\n\n"
        f"active_rubric_ids: {json.dumps(rubric_ids, ensure_ascii=False)}\n"
        "Rubrics:\n"
        f"{json.dumps(rubrics, ensure_ascii=False, indent=2)}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "overall_score": 0-1,\n'
        '  "reason": "<=60字，总体理由",\n'
        '  "rubric_results": [\n'
        "    {\n"
        '      "id": "rubric id",\n'
        '      "triggered": true,\n'
        '      "score": 0,\n'
        '      "deduction": 0-1,\n'
        '      "gate_failed": false,\n'
        '      "evidence": "<=30字",\n'
        '      "reason": "<=50字"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"{system_block}"
        f"历史轮次:\n{history_block if history_block else '<无历史轮次>'}\n\n"
        f"用户问题:\n{question}\n\n"
        f"当前候选回复_META（仅用于策略槽位一致性检查，不是用户可见回复）:\n{answer_meta_block if answer_meta_block else '<无META>'}\n\n"
        f"当前候选回复_FINAL（用户可见回复，主要评分对象）:\n{answer_final_block}\n"
    )


def _normalize_bool(value: Any, default: bool = False) -> bool:
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


def _normalize_rubric_results(obj: dict[str, Any], rubric_set: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        str(item.get("id")): item
        for item in rubric_set.get("rubrics", [])
        if isinstance(item, dict) and item.get("enabled") is not False and item.get("id")
    }
    raw_results = obj.get("rubric_results")
    if not isinstance(raw_results, list):
        raw_results = []

    seen: dict[str, dict[str, Any]] = {}
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "")).strip()
        if rid not in by_id:
            continue
        triggered = _normalize_bool(row.get("triggered"), default=True)
        try:
            raw_score = int(float(row.get("score", 1)))
        except Exception:
            raw_score = 1
        rubric_score = max(0, min(2, raw_score))
        weight = float(by_id[rid].get("weight", 0.0))
        deduction = row.get("deduction")
        if deduction is None:
            deduction = weight * (1.0 - rubric_score / 2.0) if triggered else 0.0
        try:
            deduction_f = max(0.0, float(deduction))
        except Exception:
            deduction_f = 0.0
        seen[rid] = {
            "id": rid,
            "name_cn": by_id[rid].get("name_cn"),
            "rule_type": by_id[rid].get("rule_type"),
            "weight": weight,
            "triggered": triggered,
            "score": rubric_score if triggered else None,
            "deduction": deduction_f,
            "gate_failed": _normalize_bool(row.get("gate_failed"), default=False),
            "evidence": str(row.get("evidence", ""))[:80],
            "reason": str(row.get("reason", ""))[:120],
        }

    for rid, rubric in by_id.items():
        if rid in seen:
            continue
        seen[rid] = {
            "id": rid,
            "name_cn": rubric.get("name_cn"),
            "rule_type": rubric.get("rule_type"),
            "weight": float(rubric.get("weight", 0.0)),
            "triggered": False,
            "score": None,
            "deduction": 0.0,
            "gate_failed": False,
            "evidence": "",
            "reason": "judge未返回该项，按未触发处理",
        }
    return [seen[rid] for rid in by_id]


def _semantic_score(results: list[dict[str, Any]], fallback: float) -> tuple[float, bool, str]:
    triggered = [r for r in results if r.get("triggered")]
    weighted = [r for r in triggered if float(r.get("weight", 0.0)) > 0]
    if weighted:
        denom = sum(float(r.get("weight", 0.0)) for r in weighted)
        score = sum(float(r.get("weight", 0.0)) * (float(r.get("score", 0.0)) / 2.0) for r in weighted) / denom
    else:
        score = fallback

    gate_failed = False
    gate_reason = ""
    for r in triggered:
        if r.get("rule_type") != "gate":
            continue
        if r.get("gate_failed") or float(r.get("score", 2.0)) <= 0:
            gate_failed = True
            gate_reason = str(r.get("reason", ""))[:120]
            break
    if gate_failed:
        return 0.0, True, gate_reason
    return _clip(score), False, gate_reason


async def score_output_contact_rubric(
    question: str,
    output_answer: str,
    api_base: str,
    api_key: str,
    judge_model: str,
    timeout_s: float = 45.0,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        rubric_set, rubric_path = load_active_rubric_set(
            rubric_path=kwargs.get("rubric_path"),
            rubric_index_path=kwargs.get("rubric_index_path"),
        )
        hard_config = load_hard_config(kwargs.get("hard_config_path"))
    except Exception as exc:
        return {
            "score": 0.5,
            "status": f"config_error:{type(exc).__name__}",
            "reason": str(exc)[:120],
            "raw": "",
        }

    hard = compute_hard_penalties(output_answer, hard_config)
    if not api_base or not api_key or not judge_model:
        score = _clip(0.5 - float(hard["hard_penalty_total"]))
        return {
            "score": score,
            "status": "skipped_missing_cloud_config",
            "reason": "",
            "raw": "",
            "rubric_version": rubric_set.get("version"),
            "rubric_path": str(rubric_path),
            "model_judge_score_raw": 0.5,
            "semantic_score": 0.5,
            "rubric_results": [],
            **hard,
        }

    system_prompt = str(kwargs.get("system_prompt", "")).strip()
    history_text = str(kwargs.get("history_text", "")).strip()
    answer_meta = _extract_meta_text(output_answer)
    answer_final = _extract_final_text(output_answer)
    answer_block = answer_final if answer_final else output_answer.strip()
    prompt = _build_rubric_prompt(
        question=question,
        answer_final_block=answer_block,
        answer_meta_block=answer_meta,
        history_block=history_text,
        system_prompt=system_prompt,
        rubric_set=rubric_set,
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(kwargs.get("judge_max_tokens", 1600)),
    }

    max_retries = max(1, int(kwargs.get("score_max_retries", kwargs.get("ab_max_retries", 5))))
    backoff_base_s = max(0.2, float(kwargs.get("score_backoff_base_s", kwargs.get("ab_backoff_base_s", 0.8))))
    backoff_max_s = max(backoff_base_s, float(kwargs.get("score_backoff_max_s", kwargs.get("ab_backoff_max_s", 15.0))))

    try:
        last_exc: Exception | None = None
        resp: dict[str, Any] | None = None
        for attempt in range(max_retries):
            try:
                resp = await _chat(api_base=api_base, api_key=api_key, payload=payload, timeout_s=timeout_s)
                break
            except aiohttp.ClientResponseError as exc:
                last_exc = exc
                if exc.status in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    retry_after = None
                    if exc.headers is not None:
                        ra = exc.headers.get("Retry-After")
                        if ra is not None:
                            try:
                                retry_after = float(ra)
                            except Exception:
                                retry_after = None
                    if retry_after is None:
                        retry_after = min(backoff_max_s, backoff_base_s * (2**attempt))
                    retry_after += random.uniform(0.0, 0.4)
                    await asyncio.sleep(retry_after)
                    continue
                raise
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    sleep_s = min(backoff_max_s, backoff_base_s * (2**attempt)) + random.uniform(0.0, 0.3)
                    await asyncio.sleep(sleep_s)
                    continue
                raise

        if resp is None and last_exc is not None:
            raise last_exc

        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        obj = _extract_json_dict(text)
        if obj is None:
            score = _clip(0.5 - float(hard["hard_penalty_total"]))
            return {
                "score": score,
                "status": "parse_failed",
                "reason": "",
                "raw": text,
                "rubric_version": rubric_set.get("version"),
                "rubric_path": str(rubric_path),
                "model_judge_score_raw": 0.5,
                "semantic_score": 0.5,
                "rubric_results": [],
                **hard,
            }

        try:
            fallback = _clip(float(obj.get("overall_score", 0.5)))
        except Exception:
            fallback = 0.5
        rubric_results = _normalize_rubric_results(obj, rubric_set)
        semantic, gate_failed, gate_reason = _semantic_score(rubric_results, fallback)
        final_score = _clip(semantic - float(hard["hard_penalty_total"]))

        return {
            "score": final_score,
            "status": "ok",
            "reason": str(obj.get("reason", ""))[:120],
            "raw": text,
            "rubric_version": rubric_set.get("version"),
            "rubric_path": str(rubric_path),
            "model_judge_score_raw": fallback,
            "semantic_score": semantic,
            "rubric_results": rubric_results,
            "gate_failed": gate_failed,
            "gate_reason": gate_reason,
            **hard,
        }
    except aiohttp.ClientResponseError as exc:
        return {
            "score": 0.5,
            "status": f"error:ClientResponseError:{exc.status}",
            "reason": "",
            "raw": "",
            "rubric_version": rubric_set.get("version"),
            "rubric_path": str(rubric_path),
            "rubric_results": [],
            **hard,
        }
    except asyncio.TimeoutError:
        return {
            "score": 0.5,
            "status": "error:Timeout",
            "reason": "",
            "raw": "",
            "rubric_version": rubric_set.get("version"),
            "rubric_path": str(rubric_path),
            "rubric_results": [],
            **hard,
        }
    except aiohttp.ClientConnectionError as exc:
        return {
            "score": 0.5,
            "status": f"error:ClientConnectionError:{type(exc).__name__}",
            "reason": "",
            "raw": "",
            "rubric_version": rubric_set.get("version"),
            "rubric_path": str(rubric_path),
            "rubric_results": [],
            **hard,
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "status": f"error:{type(exc).__name__}",
            "reason": "",
            "raw": "",
            "rubric_version": rubric_set.get("version"),
            "rubric_path": str(rubric_path),
            "rubric_results": [],
            **hard,
        }
