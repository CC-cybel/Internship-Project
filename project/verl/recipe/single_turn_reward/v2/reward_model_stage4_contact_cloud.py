from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import aiohttp


_CONTACT_RULE_IDS = { 3, 4, 6, 9}
_FINAL_BLOCK_RE = re.compile(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL")

# Final-text length sweet spot and mild penalty cap.
_LEN_LOWER = 90
_LEN_UPPER = 160
_LEN_TOO_SHORT = 60
_LEN_TOO_LONG = 200
_MAX_LEN_PENALTY = 0.10
_MISSING_SEP_PENALTY = 0.01
_FILLER_WORD_PENALTY = 0.10
_FILLER_WORDS = ("其实", "说明", "明白", "直接", "好的", "收到", "了解", "原来是")


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


def _extract_final_text(answer: str) -> str:
    m = _FINAL_BLOCK_RE.search(answer or "")
    if not m:
        return ""
    return m.group(1).strip()


def _final_char_len(answer: str) -> int:
    # Keep counting logic consistent with offline stats:
    # use stripped text between BEGIN_FINAL and END_FINAL, then len(chars).
    final_text = _extract_final_text(answer)
    return len(final_text)


def _length_penalty(char_len: int) -> float:
    if char_len <= 0:
        return _MAX_LEN_PENALTY
    if _LEN_LOWER <= char_len <= _LEN_UPPER:
        return 0.0

    if char_len < _LEN_LOWER:
        # Short side: linear from 0 at 90 to 0.1 at 60 and below.
        span = max(1, _LEN_LOWER - _LEN_TOO_SHORT)
        return min(_MAX_LEN_PENALTY, (max(0, _LEN_LOWER - char_len) / span) * _MAX_LEN_PENALTY)

    # Long side: linear from 0 at 160 to 0.1 at 220 and above.
    span = max(1, _LEN_TOO_LONG - _LEN_UPPER)
    return min(_MAX_LEN_PENALTY, (max(0, char_len - _LEN_UPPER) / span) * _MAX_LEN_PENALTY)


def _sep_penalty(answer: str) -> float:
    final_text = _extract_final_text(answer)
    text = final_text if final_text else (answer or "")
    return 0.0 if "<sep>" in text else _MISSING_SEP_PENALTY


def _filler_word_penalty(text: str) -> tuple[float, list[str]]:
    """机械填充词只要命中任一项，统一扣 _FILLER_WORD_PENALTY。"""
    hits = [word for word in _FILLER_WORDS if word in (text or "")]
    if not hits:
        return 0.0, []
    return _FILLER_WORD_PENALTY, hits


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
        try:
            rid = int(rule.get("rule_id"))
        except Exception:
            continue
        if rid not in _CONTACT_RULE_IDS:
            continue

        entry = {
            "rule_id": rid,
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


async def score_output_contact_only(
    question: str,
    output_answer: str,
    api_base: str,
    api_key: str,
    judge_model: str,
    timeout_s: float = 45.0,
    **kwargs: Any,
) -> dict[str, Any]:
    if not api_base or not api_key or not judge_model:
        return {
            "score": 0.5,
            "status": "skipped_missing_cloud_config",
            "reason": "",
            "raw": "",
        }

    # system_prompt 来自 extra_info.transformed_system_prompt
    system_prompt = str(kwargs.get("system_prompt", "")).strip()
    system_block = f"系统提示:\n{system_prompt}\n\n" if system_prompt else ""

    # history_text 已在 reward_function 中处理好：assistant 只保留 BEGIN_FINAL~END_FINAL
    history_text = str(kwargs.get("history_text", "")).strip()
    history_block = history_text if history_text else "<无历史轮次>"

    # output_answer 可能包含 BEGIN_FINAL~END_FINAL，统一提取
    answer_final = _extract_final_text(output_answer)
    answer_block = answer_final if answer_final else output_answer.strip()

    prompt = (
        "你是严格的医疗咨询转化评审员。\n"
        "评测目标：对当前候选回复打分。\n"
        "评分对象只能是”当前回复“，不是历史回复本身。\n"
        "本任务只训练留联阶段，我们的任务是用最好的话术留下用户的联系方式，请仅按留联阶段规则评估。\n"
        "判定标准：\n"
        "1) 重点看留联时机、留联话术、价值交换、拒绝处理，专业性。\n"
        "2) 参照bench里的标准综合输出单个分数 score，范围[0,1]。\n"
        "3)如果和患者有高价值交换，针对患者的个人情况给出了切合患者需求的价值交换，比如说打电话告诉他什么什么怎么处理这一类的，可以给满分，但是要注意，有些话比如“你平时会经常感到莫名其妙的担心、心神不宁吗？文字沟通很难准确评估你的神经受损程度，留个电话，我让老师给你做个15分钟深度评估，把具体的调节方案告诉你。”这种就属于有些死板的交换，只有干巴巴的 “留个电话做评估”。也不能给高分（<0.6分），很多回答都会符合这一点！！！像“留个电话，我让老师给你做个15分钟深度评估，顺便发些同年龄段成功的戒断案例给你参考。你看你的电话或者微信是多少？”这种福利性就会更强一些，就可以给满分。\n"
        "4)像“文字沟通很难完全掌握细节，容易产生误导。你可以留个电话，我让老师给你做个 15 分钟深度评估，把针对性的训练方法和日常干预建议直接告诉你，你也能避免盲目带孩子去跑弯路。”因为最后一句“你也能避免盲目带孩子去跑弯路”很打动人这种也属于福利性比较强的，也可以给满分。如果没有最后一句就只能是0.5分左右\n"
        "5)在索要联系方式时还问诊，给出低分，比如“留个电话，我让老师给你做个15分钟的深度评估，把调节方案直接告诉你。顺便问下，你最近睡眠质量怎么样，有没有头晕或记忆力下降的情况？”，这种就要给低分，容易分散患者注意力，降低留联成功率。\n"
        "6)如果回答中有比如说强调免费电话，免费问诊的这种也是一个加分点，可以给满分。\n"
        "7)留个你的电话，我让老师给你做一个 15 分钟的深度评估，把具体的调节方法直接告诉你，你也好转给姑母参考，避免耽误了孩子的干预黄金期。像这种就是毫无疑问的满分，因为有适度的压力，能***引起用户危机感***。\n"
        "再给你举个详细的例子，文字沟通很难捕捉到你情绪波动的细节，容易影响判断。留个电话，我让老师给你做个15分钟深度评估，把具体的减药节奏告诉你，或者你加我微信 d86833555%，这个界面一会儿会自动掉线。这是满分%，因为有强相关福利性（减肥节奏），又有适度压力（掉线）\n"
        "文字沟通很难完全捕捉你描述减肥时的语气细节，容易产生误判。<sep>留个联系方式，我让老师给你做个15分钟深度评估，把具体的调节方案告诉你。这就是人机回答，给0.4-0.6分，人机回答不是错误回答，也不是不套电话，而是福利性不强，像这种“十五分钟深度测评”看着不错，其实一点用没有，要求严格一些。\n"
        "这里咨询的人比较多，系统有时限容易掉线，你留个电话或者微信，我把更详细的减药建议发给你。<sep>除了刚才说的，你还有没有其他不舒服，比如心慌、失眠或者情绪特别低落的情况？这个有针对性福利和危机，原本可以给满分，但是最后又问问题，就只能给低分（<0.3分），具体根据其他内容给分\n"
        "8)因为我们的模型是已经经过微调了的，所以回复都不会太差， “留个电话做评估”。这种是一定会说的，但是其实空洞，没有结合患者个人情况，不要给干巴巴的回答高分（小于0.5分），结尾问问题的也给低分（<0.3分），有些回答只说“给你具体做个评估”或者“针对性做个评估”但是没说针对性或者具体在哪里，这样也不能给高分（<0.5分）。如果你觉得具体，你就在reason中用10-15字说明你觉得具体的原因，要求要针对患者咨询内容或结合具体问题，这才叫具体，否则不给高分（<0.5分）。\n"
        "9)若system_prompt中出现“（注意：35岁以上的优先要电话，35岁以下的优先要微信，若不清楚年龄，优先要微信）”，回复必须严格遵守，否则给0分。"
        "10)你还需要检查模型的thought中是否关于用户模型分析的部分的刻画是否合理，套联策略是否最优，槽位填充是否正确，以及模型的套联策略是否与用户模型中的套联策略一致，如果这部分不正确最后的总分不超过0.8分"
        "再次严肃强调，如果系统提示中有关于[留联触发与分龄策略]的说明，必须检查当前候选回复是否严格遵守，如果候选回复没有遵守规则，score直接输出0分，reason输出未遵守分龄策略规则。\n"
        "10) 输出必须为JSON，禁止额外文本。\n\n"
        "JSON schema:\n"
        "{\n"
        '  "score": 0-1,\n'
        '  "reason": "<=40字"\n'
        "}\n\n"
        f"{system_block}"
        f"历史轮次:\n{history_block}\n\n"
        f"用户问题:\n{question}\n\n"
        f"当前候选回复:\n{answer_block}\n"
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 280,
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
            return {
                "score": 0.5,
                "status": "parse_failed",
                "reason": "",
                "raw": text,
            }

        try:
            raw_score = _clip(float(obj.get("score", 0.5)))
        except Exception:
            raw_score = 0.5
        final_text = _extract_final_text(output_answer)
        final_char_len = len(final_text)
        len_pen = _length_penalty(final_char_len)
        sep_pen = _sep_penalty(output_answer)
        filler_pen, filler_hits = _filler_word_penalty(final_text)
        score = _clip(raw_score - len_pen - sep_pen - filler_pen)

        return {
            "score": score,
            "status": "ok",
            "reason": str(obj.get("reason", ""))[:120],
            "model_judge_score_raw": raw_score,
            "final_char_len": final_char_len,
            "length_penalty": len_pen,
            "sep_penalty": sep_pen,
            "filler_word_penalty": filler_pen,
            "filler_word_hits": filler_hits,
            "raw": text,
        }
    except aiohttp.ClientResponseError as exc:
        return {
            "score": 0.5,
            "status": f"error:ClientResponseError:{exc.status}",
            "reason": "",
            "raw": "",
        }
    except asyncio.TimeoutError:
        return {
            "score": 0.5,
            "status": "error:Timeout",
            "reason": "",
            "raw": "",
        }
    except aiohttp.ClientConnectionError as exc:
        return {
            "score": 0.5,
            "status": f"error:ClientConnectionError:{type(exc).__name__}",
            "reason": "",
            "raw": "",
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "status": f"error:{type(exc).__name__}",
            "reason": "",
            "raw": "",
        }
