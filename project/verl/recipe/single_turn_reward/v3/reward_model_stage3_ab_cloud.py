from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import aiohttp


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


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


def _load_bench_rules() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "bench_excellent.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _to_compact_rule_text() -> str:
    rules = _load_bench_rules()
    compact = []
    for rule in rules:
        entry = {
            "rule_id": rule.get("rule_id"),
            "rule_name_cn": rule.get("rule_name_cn"),
            "strategy": [
                {
                    "strategy_name": s.get("strategy_name"),
                    "description": s.get("description"),
                    "score": s.get("score"),
                }
                for s in rule.get("strategy", [])
                if isinstance(s, dict)
            ],
        }
        compact.append(entry)
    return json.dumps(compact, ensure_ascii=False)


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


async def compare_output_vs_gts(
    question: str,
    output_answer: str,
    gts_answer: str,
    api_base: str,
    api_key: str,
    judge_model: str,
    timeout_s: float = 45.0,
    **kwargs: Any,
) -> dict[str, Any]:
    if not api_base or not api_key or not judge_model:
        return {
            "score": 0.5,
            "winner": "tie",
            "score_output": 0.5,
            "score_gts": 0.5,
            "status": "skipped_missing_cloud_config",
            "reason": "",
            "raw": "",
        }

    rules_text = _to_compact_rule_text()
    history_text = str(kwargs.get("history_text", "")).strip()
    history_block = history_text if history_text else "<无历史轮次>"

    prompt = (
        "你是严格的医疗咨询转化A/B评测裁判。\n"
        "评测目标：候选A(模型输出) vs 候选B(gts) 谁更好。\n"
        "评分对象只能是“当前回复”，不是历史回复本身。\n"
        "评分必须严格遵循下述 bench 规则。\n\n"
        "bench规则(JSON压缩)：\n"
        f"{rules_text}\n\n"
        "判定标准：\n"
        "1) 重点看问诊逻辑、推进性、留联时机、留联话术、拒绝处理、专业性。\n"
        "2) 输出 winner 仅能是 A/B/tie。\n"
        "3) 分别给 score_a 和 score_b，范围[0,1]。\n"
        "4) 如两者优劣不明显，winner=tie。\n"
        "5) 输出必须为JSON，禁止额外文本。\n\n"
        "JSON schema:\n"
        "{\n"
        '  "winner": "A|B|tie",\n'
        '  "score_a": 0-1,\n'
        '  "score_b": 0-1,\n'
        '  "reason": "<=60字"\n'
        "}\n\n"
        f"历史轮次:\n{history_block}\n\n"
        f"用户问题:\n{question}\n\n"
        f"候选A(模型output):\n{output_answer}\n\n"
        f"候选B(gts):\n{gts_answer}\n"
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 280,
    }

    max_retries = max(1, int(kwargs.get("ab_max_retries", 5)))
    backoff_base_s = max(0.2, float(kwargs.get("ab_backoff_base_s", 0.8)))
    backoff_max_s = max(backoff_base_s, float(kwargs.get("ab_backoff_max_s", 15.0)))

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
            return {
                "score": 0.5,
                "winner": "tie",
                "score_output": 0.5,
                "score_gts": 0.5,
                "status": "parse_failed",
                "reason": "",
                "raw": text,
            }

        winner = str(obj.get("winner", "tie")).strip().lower()
        if winner not in {"a", "b", "tie"}:
            winner = "tie"

        try:
            score_a = _clip(float(obj.get("score_a", 0.5)))
        except Exception:
            score_a = 0.5
        try:
            score_b = _clip(float(obj.get("score_b", 0.5)))
        except Exception:
            score_b = 0.5

        # If A/B scores are close enough, treat as tie in reward.
        if abs(score_a - score_b) <= 0.1:
            reward = 0.5
        elif winner == "a":
            reward = 1.0
        elif winner == "b":
            reward = 0.0
        else:
            reward = 0.5

        return {
            "score": reward,
            "winner": winner,
            "score_output": score_a,
            "score_gts": score_b,
            "status": "ok",
            "reason": str(obj.get("reason", ""))[:120],
            "raw": text,
        }
    except aiohttp.ClientResponseError as exc:
        return {
            "score": 0.5,
            "winner": "tie",
            "score_output": 0.5,
            "score_gts": 0.5,
            "status": f"error:ClientResponseError:{exc.status}",
            "reason": "",
            "raw": "",
        }
    except asyncio.TimeoutError:
        return {
            "score": 0.5,
            "winner": "tie",
            "score_output": 0.5,
            "score_gts": 0.5,
            "status": "error:Timeout",
            "reason": "",
            "raw": "",
        }
    except aiohttp.ClientConnectionError as exc:
        return {
            "score": 0.5,
            "winner": "tie",
            "score_output": 0.5,
            "score_gts": 0.5,
            "status": f"error:ClientConnectionError:{type(exc).__name__}",
            "reason": "",
            "raw": "",
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "winner": "tie",
            "score_output": 0.5,
            "score_gts": 0.5,
            "status": f"error:{type(exc).__name__}",
            "reason": "",
            "raw": "",
        }
