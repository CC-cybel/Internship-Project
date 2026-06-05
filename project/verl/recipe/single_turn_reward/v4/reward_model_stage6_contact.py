from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

import aiohttp


_FINAL_BLOCK_RE = re.compile(r"BEGIN_FINAL\s*\n?([\s\S]*?)\n?END_FINAL")


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
        pass
    return None


def _extract_final_text(answer: str) -> str:
    m = _FINAL_BLOCK_RE.search(answer or "")
    if not m:
        return ""
    return m.group(1).strip()


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


def _history_text(extra_info: dict[str, Any] | None, max_turns: int = 16) -> str:
    extra_info = extra_info or {}
    turns = None
    for key in ("history", "conversations", "dialogue"):
        value = extra_info.get(key)
        if isinstance(value, list):
            turns = value
            break
    if not turns:
        return ""
    normalized = []
    for t in turns:
        item = _normalize_turn(t)
        if item is not None:
            normalized.append(item)
    if not normalized:
        return ""
    clipped = normalized[-max_turns:]
    return "\n".join(f"{r}: {c}" for r, c in clipped)


_JUDGE_PROMPT_TEMPLATE = """你是严格的医疗咨询转化评审员，专注于评估留联阶段的策略执行质量。

【任务】
你将收到：
  1. 系统提示词（System Prompt）—— 定义了该样本的留联策略规则
  2. 历史对话
  3. 当前轮次信息
  4. 客服模型的当前回复

你的职责：根据系统提示词中定义的规则，对客服回复打分。

【评分标准】
score = 1：客服在当前轮次正确执行了留联策略。
score = 0：客服未正确执行留联策略（包括未索要联系方式、方式错误、渠道错误等）。

【具体判断步骤】
Step 1 — 判断是否应触发留联：
  - 若 contact_round > 0 且 turn_round >= contact_round，则本轮需要执行留联。
  - 若未触发留联轮次，score = 1（因为无需留联）。

Step 2 — 若触发留联，判断回复质量：
  从系统提示词中提取并综合以下规则：
    (a) 分龄定向策略（rule_id=15）：年龄大于35岁→索要电话；年龄≤35岁或未知→索要微信
    (b) 留联优先级（rule_id=14）：默认优先微信，降级才要电话
    (c) 意向套电（rule_id=13）：用户表达留联意愿时立即套电
  根据系统提示词综合以上规则，判断客服回复中索要联系方式的方式是否符合策略。
  正确执行 = 用了正确的渠道 + 没有出现错误渠道的关键词。
  若回复中同时提及电话和微信，以优先渠道为准；若优先渠道错误，直接判0分。

Step 3 — 若留联已成功触发但回复未提供任何联系方式，判0分。

【输出格式】（严格 JSON，禁止额外文本）
{{
  "contact_triggered": true或false,
  "inferred_user_age": "数字或unknown",
  "contact_triggered_reason": "留联是否触发的判断理由",
  "contact_method_used": "phone或wechat或none",
  "score": 0或1,
  "reason": "不超过100字的简短理由"
}}

====================================================
【系统提示词】
{system_prompt}
====================================================

【历史对话】
{history_block}

【当前轮次】
turn_round: {turn_round}
contact_round: {contact_round}

【客服当前回复】
{output_answer}
"""


async def _chat(
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: float = 45.0,
) -> dict[str, Any]:
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


async def score_output_contact_stage(
    question: str,
    output_answer: str,
    api_base: str,
    api_key: str,
    judge_model: str,
    timeout_s: float = 45.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    留联阶段奖励评分（LLM-as-judge）。

    核心流程：
      1. 判断 turn_round >= contact_round → 需要触发留联
      2. 把 system_prompt + history + output_answer 全部发给 judge
      3. judge 从 system_prompt 中解析规则并打分
    """
    if not api_base or not api_key or not judge_model:
        return {
            "score": 0.5,
            "status": "skipped_missing_cloud_config",
            "reason": "",
            "raw": "",
        }

    system_prompt = str(kwargs.get("system_prompt", "")).strip()
    history_text = str(kwargs.get("history_text", "")).strip()
    turn_round = int(_to_float(kwargs.get("turn_round"), 0))
    contact_round = int(_to_float(kwargs.get("contact_round"), 0))

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt or "<无系统提示词>",
        history_block=history_text or "<无历史轮次>",
        turn_round=turn_round,
        contact_round=contact_round,
        output_answer=output_answer,
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 400,
    }

    max_retries = max(1, int(kwargs.get("score_max_retries", 5)))
    backoff_base_s = max(0.2, float(kwargs.get("score_backoff_base_s", 0.8)))
    backoff_max_s = max(backoff_base_s, float(kwargs.get("score_backoff_max_s", 15.0)))

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
                await asyncio.sleep(
                    min(backoff_max_s, backoff_base_s * (2**attempt)) + random.uniform(0.0, 0.3)
                )
                continue
            raise

    if resp is None and last_exc is not None:
        raise last_exc  # type: ignore

    raw_text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    obj = _extract_json_dict(raw_text)
    if obj is None:
        return {
            "score": 0.5,
            "status": "parse_failed",
            "reason": "",
            "raw": raw_text,
        }

    score = _clip(_to_float(obj.get("score"), 0.5))
    return {
        "score": score,
        "status": "ok",
        "contact_triggered": bool(obj.get("contact_triggered", False)),
        "inferred_user_age": str(obj.get("inferred_user_age", "unknown")),
        "contact_method_used": str(obj.get("contact_method_used", "none")),
        "contact_triggered_reason": str(obj.get("contact_triggered_reason", ""))[:120],
        "reason": str(obj.get("reason", ""))[:120],
        "raw": raw_text,
    }
