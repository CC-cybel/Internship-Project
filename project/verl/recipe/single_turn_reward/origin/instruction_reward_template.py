"""Template for instruction-following reward collaborator."""

from __future__ import annotations

import json
import re
from typing import Any

import aiohttp

# Fill this prompt by instruction-following reward owner.
INSTRUCTION_PROMPT_TEMPLATE = ""


def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _parse_score(raw: str, score_max: float = 10.0) -> float:
    raw = (raw or "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            return _clip(float(data["score"]) / score_max)
    except Exception:
        pass
    m = re.search(r"[-+]?\d*\.?\d+", raw)
    if not m:
        return 0.0
    return _clip(float(m.group(0)) / score_max)


async def _chat_complete(
    reward_router_address: str,
    payload: dict[str, Any],
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    url = f"http://{reward_router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def compute_instruction_score(
    question: str,
    answer: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return {'score': float in [0,1], 'raw': str, 'status': str}."""
    del ground_truth
    del extra_info

    if not INSTRUCTION_PROMPT_TEMPLATE.strip():
        return {"score": 0.0, "raw": "", "status": "instruction_prompt_empty"}
    if not reward_router_address:
        return {"score": 0.0, "raw": "", "status": "reward_router_address_missing"}

    judge_model = kwargs.get("instruction_judge_model") or kwargs.get("judge_model")
    if not judge_model:
        return {"score": 0.0, "raw": "", "status": "judge_model_missing"}

    prompt = INSTRUCTION_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        extra_info_json=json.dumps(extra_info, ensure_ascii=False),
    )
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(kwargs.get("instruction_judge_temperature", 0.0)),
        "top_p": float(kwargs.get("instruction_judge_top_p", 1.0)),
        "max_tokens": int(kwargs.get("instruction_judge_max_tokens", 64)),
    }

    try:
        resp = await _chat_complete(
            reward_router_address=reward_router_address,
            payload=payload,
            timeout_s=float(kwargs.get("judge_timeout_s", 60.0)),
        )
        raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        score = _parse_score(raw, score_max=float(kwargs.get("instruction_judge_score_max", 10.0)))
        return {"score": score, "raw": raw, "status": "ok"}
    except Exception as exc:
        return {"score": 0.0, "raw": "", "status": f"instruction_error:{type(exc).__name__}"}

