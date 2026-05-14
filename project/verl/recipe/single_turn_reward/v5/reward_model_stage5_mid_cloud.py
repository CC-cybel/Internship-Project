from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import aiohttp


_MID_RULE_IDS = {1, 5, 7, 8, 9, 12, 13}
_FINAL_BLOCK_RE = re.compile(r"BEGIN_FINAL\s*\n([\s\S]*?)\nEND_FINAL")

_LEN_LOWER = 0
_LEN_UPPER = 66
_LEN_TOO_SHORT = 0
_LEN_TOO_LONG = 100
_MAX_LEN_PENALTY = 0.10
_MAX_QUESTION_PENALTY = 0.10
_SEP_PENALTY = 0.10
_FILLER_WORD_PENALTY = 0.10
_FILLER_WORDS = ("其实", "说明", "明白", "直接", "好的", "收到", "了解", "原来是")
_MAX_DIAGNOSIS_PENALTY = 0.30


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


def _final_char_len(answer: str) -> int:
    final_text = _extract_final_text(answer)
    return len(final_text)


def _length_penalty(char_len: int) -> float:
    """超过 _LEN_UPPER 时每个字符惩罚 0.01，上限 _MAX_LEN_PENALTY。"""
    if char_len <= _LEN_UPPER:
        return 0.0
    return min(_MAX_LEN_PENALTY, (char_len - _LEN_UPPER) * 0.005)


def _question_penalty(text: str) -> float:
    """超出1个问号后每个问号惩罚 0.1，上限 _MAX_QUESTION_PENALTY。"""
    count = text.count("？") + text.count("?")
    if count <= 1:
        return 0.0
    return min(_MAX_QUESTION_PENALTY, (count - 1) * 0.1)


def _sep_penalty(text: str) -> float:
    """
    BEGIN_FINAL 内容超过 76 字符且未使用 <sep> 分句时惩罚 _SEP_PENALTY。
    """
    if len(text) <= 76:
        return 0.0
    if "<sep>" in text or "<SEP>" in text:
        return 0.0
    return _SEP_PENALTY


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
        if rid not in _MID_RULE_IDS:
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


_JUDGE_PROMPT_TEMPLATE = """你是严格的医疗咨询转化评审员，专门评估 mid 阶段（价值建立期）的回复质量。
评测目标：对当前候选回复打分。
评分对象只能是"当前回复"，不是历史回复本身。
本阶段核心任务：推进对话 + 深化价值锚定 + 埋下留联钩子（不执行留联）。

【评分维度】（请综合评判输出单个分数 score，范围[0,1]）

一、价值建立
   高分表现：
     1. 紧密结合用户已提供的具体信息（年龄/症状/病程/身份/既往史）进行分析。
     2. 解释症状背后的生理/心理机制，让用户感觉"被专业理解"
     3. 指出潜在风险或长期危害，制造适度的紧迫感（但不恐吓）
     4. 答疑后问诊，主动推进对话，让用户有话可说。
     5. 示例1：初三学生+两个多月+睡9小时仍困 -> 关联学业压力、睡眠效率、神经发育特点。
     6. 示例2：57岁女性+几十年失眠 -> 关联更年期神经调节、长期神经记忆固化
     7. 示例3：30岁男性+同性恋 -> 强调共情，同性恋心理健康问题。

   扣分表现：
     1. 泛泛而谈，没有结合用户具体情况（通用模板），给0.5分以下。
     2. 只解释不推进，停留在表面安慰，表现为没有提问，给0.5分以下
     3. 常识错误：用户身份识别错误、性别错误（如没说自己性别主观带入女性）、逻辑错误（如"在您不方便的时候联系您"），直接给0分。

二、已知信息与追问的区分（核心判断规则）
   以下情况才构成"已知信息重复提问"——给0分：
     1. 用户本人明确说过"年龄"，模型再问年龄
     2. 用户本人明确说过"性别"，模型再问性别
     3. 用户刚说过某个具体症状，模型又重复问这个症状
   特别说明——主动询问基本信息是正常行为，不扣分：
     - 如果模型还没有拿到用户的年龄/性别等基本信息，主动询问是应当的，不属于"重复提问"
     - 只有在用户已经明确说过某信息后，模型仍然再问，才算重复提问
   判断方法：先看用户当前问题，再看历史中用户是否已提供过该信息。

【打分要求】
  1. 先审核严重错误：身份/性别识别出错或无法推断、逻辑错误（0分）；已知信息直接重复提问（0分）。有时年龄和性别不是必须要明确进行询问才能知道，有时出现一些和年龄，性别关联较强的对话你可以合理推测填入，比如说自己高三，这就可以推测出咨询者为本人，年龄可以填18岁，比如说胸部发育，可以推测性别为女性，比如说阴茎，睾丸，那就是男性。比如说自己老婆，那咨询者就是老公，等等。如果出错就直接0分。
  2. 再审核中等错误：通用模板无针对性、推进不够自然、停留在表面安慰（0.5-0.6分）。
  3. 最后审核优秀表现：结合用户具体情况+专业分析+主动收集基本信息（如未获取）+自然推进对话（0.9-1.0分）。
  4. score 只评价 mid 阶段回复质量，不要把下方“诊断类独立惩罚”混入 score。

【诊断类独立惩罚】
仅当当前回复正在做诊断、病因判断、疾病归因或病情分析时适用；如果当前回复没有进入诊断分析，diagnosis_penalty 必须为 0。
请独立判断以下三条规则，每命中一条惩罚 0.1，最多 0.3。该惩罚必须与 score 完全独立，不要因为输出惩罚而调整 score。
  1. 无依据诊断：当前回复凭空猜测病因，缺乏逻辑支撑。
  2. 诊断分析错误：当前回复诊断分析出现错误。
  3. 唯一性诊断：当前回复仅凭简单描述即断定为某种疾病，未给可能性或排除性建议。
  "trigger_condition": "只有在当前回复在诊断的时候才适用"
【输出格式】（必须严格遵循）
输出必须为 JSON，禁止任何额外文本。
JSON schema:
{{
  "score": 0-1之间的单个分数,
  "reason": "不超过100字的简短理由，包含你认为的已知信息",
  "diagnosis_penalty": 0到0.3之间的独立惩罚分,
  "diagnosis_penalty_rules": ["命中的规则名，如无则为空数组"],
  "diagnosis_penalty_reason": "不超过60字；无惩罚则写无"
}}

【评分示例】
- 优秀（0.9-1.0）：主动收集用户基本信息+紧密结合具体信息+深入分析+推进对话自然流畅
- 良好（0.7-0.85）：基本符合要求，有一定针对性，轻微模板化或推进不够自然
- 一般（0.5-0.7）：有模板化倾向或轻微问题，存在通用回复等
- 差（0-0.5）：身份识别错误、逻辑错误、直接重复用户已说过的信息

历史轮次:
{history_block}

系统提示:
{system_block}

当前轮次: {turn_round}
留联触发轮次: {contact_round}

用户当前问题:
{question}

当前助手回复:
{output_answer}
"""


async def score_output_mid_stage(
    question: str,
    output_answer: str,
    api_base: str,
    api_key: str,
    judge_model: str,
    timeout_s: float = 45.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    中间阶段（mid-stage）奖励评分函数。

    目标阶段：start 之后、contact 之前（通常第3轮到留联触发前1轮）
    核心任务：
      1) 深化价值建立 - 结合用户具体情况分析，建立"必须专业干预"的认知
      2) 巩固专业信任 - 体现医学专业性，让用户信服
      3) 推进对话 - 持续挖掘信息，保持对话动力
      4) 自然铺垫留联 - 暗示需要专家评估/电话沟通，但不执行留联

    评分维度（多维度综合）：
      - stage_fit: 回复是否符合mid阶段定位（非start也非contact）
      - value_building: 价值建立质量（是否结合具体情况，是否制造紧迫感）
      - trust_building: 信任建立（专业性、共情、非模板化）
      - progress: 对话推进（是否获取新信息/明确下一步）
      - hook_quality: 留联铺垫质量（是否为后续留联埋下伏笔）
    """
    if not api_base or not api_key or not judge_model:
        return {
            "score": 0.5,
            "status": "skipped_missing_cloud_config",
            "raw": "",
            "stage": "mid",
        }

    history_text = str(kwargs.get("history_text", "")).strip()
    turn_round = int(_to_float(kwargs.get("turn_round"), 0))
    contact_round = int(_to_float(kwargs.get("contact_round"), 0))

    # 阶段判断（数据已筛选过，此处为兜底）
    is_start_stage = turn_round <= 2
    is_contact_stage = contact_round > 0 and turn_round >= contact_round

    if is_start_stage or is_contact_stage:
        stage_label = "contact" if is_contact_stage else "start"
        return {
            "score": 0.5,
            "status": "skipped_wrong_stage",
            "reason": f"not_mid_stage: turn_round={turn_round}, contact_round={contact_round}, stage={stage_label}",
            "raw": "",
            "stage": stage_label,
        }

    bench_reference = _to_compact_rule_text()
    system_prompt = str(kwargs.get("system_prompt", "")).strip()
    system_block = system_prompt if system_prompt else "<无系统提示>"
    history_block = history_text if history_text else "<无历史轮次>"

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        history_block=history_block,
        system_block=system_block,
        turn_round=turn_round,
        contact_round=contact_round,
        question=question,
        output_answer=output_answer,
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 320,
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

        score = _clip(_to_float(obj.get("score"), 0.5))

        # 长度惩罚：对 BEGIN_FINAL 提取的纯回复文本计算
        final_text = _extract_final_text(output_answer)
        final_char_len = len(final_text)
        len_pen = _length_penalty(final_char_len)

        # 问句过多惩罚：对纯回复文本计算问号数量
        q_pen = _question_penalty(final_text)

        # 分句惩罚：超过 76 字符且未使用 <sep> 分句
        sep_pen = _sep_penalty(final_text)

        # 机械填充词惩罚：命中任一填充词统一扣 0.1
        filler_pen, filler_hits = _filler_word_penalty(final_text)

        # 诊断类独立惩罚：由 judge 单独输出，不混入 judge score，再由本地统一扣除
        diagnosis_pen = _clip(_to_float(obj.get("diagnosis_penalty"), 0.0), 0.0, _MAX_DIAGNOSIS_PENALTY)
        diagnosis_rules = obj.get("diagnosis_penalty_rules", [])
        if not isinstance(diagnosis_rules, list):
            diagnosis_rules = []

        score = _clip(score - len_pen - q_pen - sep_pen - filler_pen - diagnosis_pen)

        return {
            "score": score,
            "status": "ok",
            "reason": str(obj.get("reason", ""))[:120],
            "stage": "mid",
            "final_char_len": final_char_len,
            "length_penalty": len_pen,
            "question_penalty": q_pen,
            "sep_penalty": sep_pen,
            "filler_word_penalty": filler_pen,
            "filler_word_hits": filler_hits,
            "diagnosis_penalty": diagnosis_pen,
            "diagnosis_penalty_rules": [str(x) for x in diagnosis_rules][:3],
            "diagnosis_penalty_reason": str(obj.get("diagnosis_penalty_reason", ""))[:80],
            "model_judge_score_raw": _clip(_to_float(obj.get("score"), 0.5)),
            "raw": text,
        }
    except aiohttp.ClientResponseError as exc:
        msg = (exc.message or "").strip().replace("\n", " ")
        if len(msg) > 120:
            msg = msg[:120]
        return {
            "score": 0.5,
            "status": f"error:ClientResponseError:{exc.status}:{msg}",
            "raw": "",
            "stage": "mid",
        }
    except asyncio.TimeoutError:
        return {
            "score": 0.5,
            "status": "error:Timeout",
            "raw": "",
            "stage": "mid",
        }
    except aiohttp.ClientConnectionError as exc:
        return {
            "score": 0.5,
            "status": f"error:ClientConnectionError:{type(exc).__name__}",
            "raw": "",
            "stage": "mid",
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "status": f"error:{type(exc).__name__}",
            "raw": "",
            "stage": "mid",
        }
