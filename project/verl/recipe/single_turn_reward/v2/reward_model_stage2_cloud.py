from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import aiohttp


_AI_STYLE_KEYWORDS = (
    "为了",
    "直接",
    "请问有什么可以帮您",
    "理解",
    "抱歉",
)
_TEXT_NOISE_RE = re.compile(r"[\s\u3000\.,，。!！\?？:：;；\-—_~`'\"“”‘’()（）\[\]【】<>《》/\\|]+")
_BENCH_CACHE: dict[int, dict[str, Any]] | None = None
_STAGE_RULE_IDS = {
    "start": [1, 7, 8, 12],
    "mid": [1, 5, 7, 8, 9, 12],
    "contact": [2, 3, 4, 6, 9, 10, 11],
}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_first_float(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


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

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _style_penalty(text: str) -> float:
    hits = _count_style_keyword_hits(text)
    if hits <= 0:
        return 0.0
    return min(0.2, 0.06 * hits)


def _normalize_for_keyword_match(text: str) -> str:
    text = (text or "").lower()
    return _TEXT_NOISE_RE.sub("", text)


def _count_style_keyword_hits(text: str) -> int:
    normalized = _normalize_for_keyword_match(text)
    if not normalized:
        return 0

    hits = 0
    for kw in _AI_STYLE_KEYWORDS:
        nkw = _normalize_for_keyword_match(kw)
        if not nkw:
            continue
        start = 0
        while True:
            idx = normalized.find(nkw, start)
            if idx < 0:
                break
            hits += 1
            start = idx + len(nkw)
    return hits


def _load_bench_rules() -> dict[int, dict[str, Any]]:
    global _BENCH_CACHE
    if _BENCH_CACHE is not None:
        return _BENCH_CACHE

    path = Path(__file__).with_name("bench_excellent.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: dict[int, dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                rid = int(item.get("rule_id"))
            except Exception:
                continue
            out[rid] = item

    _BENCH_CACHE = out
    return out


def _infer_stage(turn_round: int, contact_round: int) -> str:
    if turn_round <= 2:
        return "start"
    if contact_round > 0 and turn_round >= contact_round:
        return "contact"
    return "mid"


def _build_bench_reference(stage: str) -> str:
    rules = _load_bench_rules()
    ids = _STAGE_RULE_IDS.get(stage, _STAGE_RULE_IDS["mid"])

    blocks: list[str] = []
    for rid in ids:
        rule = rules.get(rid)
        if not rule:
            continue

        title = f"rule_id={rid} {rule.get('rule_name_cn', '')}/{rule.get('rule_name', '')}".strip()
        lines = [title]

        strategies = rule.get("strategy")
        if isinstance(strategies, list):
            for s in strategies[:4]:
                if not isinstance(s, dict):
                    continue
                sname = str(s.get("strategy_name", "")).strip()
                sdesc = str(s.get("description", "")).strip().replace("\n", " ")
                sscore = s.get("score", "")
                lines.append(f"- {sname} (score={sscore}): {sdesc}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else "<bench规则缺失>"


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
                # Keep server message so caller can record the root cause in status.
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=text[:300],
                    headers=resp.headers,
                )
            return json.loads(text)


async def compute_model_style_score(
    question: str,
    answer: str,
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
            "raw": "",
            "stage": "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }

    history_text = str(kwargs.get("history_text", "")).strip()
    turn_round = int(_to_float(kwargs.get("turn_round"), 0))
    contact_round = int(_to_float(kwargs.get("contact_round"), 0))
    force_start = turn_round <= 2
    expected_stage_hint = str(kwargs.get("expected_stage_hint", "")).strip().lower()
    if expected_stage_hint not in {"start", "mid", "contact"}:
        expected_stage_hint = "unknown"
    if force_start:
        expected_stage_hint = "start"
    inferred_stage = _infer_stage(turn_round=turn_round, contact_round=contact_round)
    active_stage = expected_stage_hint if expected_stage_hint in {"start", "mid", "contact"} else inferred_stage
    if force_start:
        active_stage = "start"
    is_start_stage = active_stage == "start"
    bench_reference = _build_bench_reference(active_stage)

    history_block = history_text if history_text else "<无历史轮次>"

    prompt = (
        "你是医疗咨询转化场景的严格评审员，这个场景是线上和患者对话，以获取患者的联系方式，主要核心是进行联系方式的获取，而非问诊，所有话术均服务于留联目的。\n"
        "你的任务是基于历史对话和当前问题对用户的回答进行评分，评分对象只能是“当前回复”（Action + Content），不是历史回复本身。\n"
        "请先根据当前回复判定阶段，再按阶段细则给分。\n"
        "阶段仅允许: start/mid/contact。\n"
        "全局维度定义:\n"
        "1) stage_fit: 当前回复与判定阶段是否一致。\n"
        "2) objective: 当前阶段目标达成度。具体而言在start阶段体现为是否获取新的有效信息，是否推进下一问题。在mid阶段体现为是否建立问题严重性 ，是否推动对话 ，是否铺垫留联必要性。在contact阶段体现为是否高福利性地提出联系方式 ，是否给价值交换，是否结合具体情况。\n"
        "3) professional: 专业性与沟通质量。\n"
        "4) benchmark_alignment: 与对应阶段bench规则的一致性。\n"
        "\n"
        "阶段细则(参考优秀评估规则):\n"
        "A. start 阶段(开始问诊)\n"
        "- 高分行为: 主诉优先, 对症追问, 由浅入深(主诉->细节->影响), 每轮都要推进下一步。\n"
        "- 扣分行为: 无效问诊(离题/查户口/重复问已给信息), 空泛问诊(泛泛一句无诊断价值), 只回答不推进。\n"
        "- 关键要求: 必须有推进动作(明确下一问题/下一步检查点/过渡到价值建立)。\n"
        "- 红线: 绝对化威胁、强制干扰、读起来让人感觉绝对不适。\n"
        "\n"
        "B. mid 阶段(价值建立)\n"
        "- 高分行为: 给希望+给方向(解释原因/风险趋势/可改善路径), ***有意识的结合对方实际情况建立价值，推动对话***并自然过渡到留联必要性。\n"
        "- 扣分行为: 直接给完整终局解决方案(让用户无需留联), 机械复读、泛泛安慰、只解释不推进。\n"
        "- 关键要求: mid 的目标是'建立必要性并推进留联'，不是在当前轮把问题彻底解决。\n"
        "- 红线: 辩论压制用户、冷漠忽视痛苦、虚假承诺。\n"
        "\n"
        "C. contact 阶段(留联收口)\n"
        "- 高分行为: 时机自然(价值前置或共情后), 给出明确价值交换(方案/资料/评估), 可使用礼貌的时机压力。\n"
        "- 礼貌时机压力(可加分): 例如提醒'延误可能增加处理成本/错过更佳干预窗口'，语气克制且基于已知信息。\n"
        "- 扣分行为: 指令式索联、乞求式索联、无价值硬要、用户拒绝后无转圜动作。\n"
        "- 红线: 绝对化威胁、强制干扰、对拒绝进行质问压迫。读起来让人感觉绝对不适。\n"
        "\n"
        "阶段与bench规则映射（参考）：\n"
        "- start 对应 rule_id: [1,7,8,12]\n"
        "- mid 对应 rule_id: [1,5,7,8,9,12]\n"
        "- contact 对应 rule_id: [2,3,4,6,9,10,11]\n"
        "***打分需要有区分度，请根据细则进行综合评判，不要过于保守或激进，对于contact阶段的评分要尤其严谨，因为这是最重要的一个环节，合理利用0-1的分数区间。***\n"
        "\n"
        "flag 判定口径（必须严格执行）：\n"
        "1) template_tone=true 仅在以下情况触发：\n"
        "   - 话术明显模板化，放到多数用户场景都几乎可原样复用；\n"
        "   - 出现明显客服套话/空泛安抚，和用户当前信息结合很弱；\n"
        "   - 连续两句以上为通用表达，缺少针对用户已提供事实的定制回应。\n"
        "2) template_tone=false 在以下情况成立：\n"
        "   - 明确引用用户本轮或历史中的具体事实（年龄/病程/症状/既往信息）并据此推进；\n"
        "   - 虽然语气稳健，但内容具有场景针对性，不是通用客服模板。\n"
        "3) no_progress=true 仅在以下情况触发：\n"
        "   - 没有新增有效信息采集，也没有推进下一步动作（问诊推进/价值建立/留联推进）；\n"
        "   - 基本在复述或泛泛回应，用户状态与对话目标没有实质前进。\n"
        "4) no_progress=false 在以下情况成立：\n"
        "   - 新获取了有效信息，或明确推进了下一步（例如更具体追问、价值锚定、自然留联动作）。\n"
        f"当前应重点参考阶段: {active_stage}\n"
        "以下是从 bench_excellent.json 提取的对应规则片段（必须据此判分）：\n"
        f"{bench_reference}\n"
        "\n"
        "输出必须为 JSON，禁止任何额外文本。\n"
        "JSON schema:\n"
        "{\n"
        '  "stage": "start|mid|contact",\n'
        '  "scores": {\n'
        '    "stage_fit": 0-1,\n'
        '    "objective": 0-1,\n'
        '    "professional": 0-1,\n'
        '    "benchmark_alignment": 0-1\n'
        "  },\n"
        '  "flags": {\n'
        '    "premature_contact": true/false,\n'
        '    "hard_sell": true/false,\n'
        '    "redline": true/false,\n'
        '    "invalid_inquiry": true/false,\n'
        '    "full_solution_too_early": true/false,\n'
        '    "template_tone": true/false,\n'
        '    "no_progress": true/false\n'
        "  },\n"
        '  "reason": "<=30字"\n'
        "}\n"
        "\n"
        f"历史轮次:\n{history_block}\n\n"
        f"当前轮次: {turn_round}\n"
        f"参考留联轮次: {contact_round}\n"
        f"规则侧阶段提示: {expected_stage_hint}\n\n"
        f"当前用户问题:\n{question}\n\n"
        f"当前助手回复:\n{answer}\n"
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 320,
    }

    try:
        last_exc: Exception | None = None
        resp: dict[str, Any] | None = None
        for attempt in range(3):
            try:
                resp = await _chat(api_base=api_base, api_key=api_key, payload=payload, timeout_s=timeout_s)
                break
            except aiohttp.ClientResponseError as exc:
                last_exc = exc
                if exc.status in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < 2:
                    await asyncio.sleep(0.8 * (attempt + 1))
                    continue
                raise
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(0.8 * (attempt + 1))
                    continue
                raise
        if resp is None and last_exc is not None:
            raise last_exc

        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        obj = _extract_json_dict(text)

        if obj is not None:
            scores = obj.get("scores", {}) if isinstance(obj.get("scores"), dict) else {}
            flags = obj.get("flags", {}) if isinstance(obj.get("flags"), dict) else {}

            stage_fit = _clip(_to_float(scores.get("stage_fit"), 0.5))
            objective = _clip(_to_float(scores.get("objective"), 0.5))
            professional = _clip(_to_float(scores.get("professional"), 0.5))
            # safety 维度不再参与评分，保持中性值用于兼容下游日志字段。
            safety = 0.5
            benchmark = _clip(_to_float(scores.get("benchmark_alignment"), 0.5))

            redline = bool(flags.get("redline", False))
            invalid_inquiry = bool(flags.get("invalid_inquiry", False))
            full_solution_too_early = bool(flags.get("full_solution_too_early", False))
            template_tone = bool(flags.get("template_tone", False))
            no_progress = bool(flags.get("no_progress", False))

            sw = _to_float(kwargs.get("stage_weight"), 0.25)
            ow = _to_float(kwargs.get("objective_weight"), 0.35)
            pw = _to_float(kwargs.get("professional_weight"), 0.15)
            bw = _to_float(kwargs.get("benchmark_weight"), 0.15)
            wsum = max(1e-6, sw + ow + pw + bw)

            score = _clip(
                (sw * stage_fit + ow * objective + pw * professional + bw * benchmark) / wsum
            )

            if force_start:
                # Early rounds are highly template-driven by prompt constraints.
                # Keep reward signal stable while still allowing variation above the floor.
                stage_fit = 1.0
                score_floor = 0.50 if turn_round == 1 else 0.45
                score = max(score, score_floor)

            style_hits = _count_style_keyword_hits(answer)

            # Start阶段不进行风格/模板/无推进三项扣分。
            if is_start_stage:
                style_pen = 0.0
            else:
                style_pen = _style_penalty(answer)
                score = _clip(score - style_pen)

                # Explicit penalties: keep sub-scores unchanged, only adjust final score.
                if template_tone:
                    score = _clip(score - 0.10)
                if no_progress:
                    score = _clip(score - 0.05)

            if force_start:
                score_floor = 0.50 if turn_round == 1 else 0.45
                score = max(score, score_floor)

            if redline:
                score = min(score, 0.2)

            stage = str(obj.get("stage", "unknown")).strip().lower()
            if stage not in {"start", "mid", "contact"}:
                stage = "unknown"
            if force_start:
                stage = "start"

            return {
                "score": score,
                "status": "ok",
                "raw": text,
                "stage": stage,
                "stage_fit_score": stage_fit,
                "objective_score": objective,
                "professional_score": professional,
                "safety_score": safety,
                "benchmark_alignment_score": benchmark,
                "premature_contact": bool(flags.get("premature_contact", False)),
                "hard_sell": bool(flags.get("hard_sell", False)),
                "redline": redline,
                "invalid_inquiry": invalid_inquiry,
                "full_solution_too_early": full_solution_too_early,
                "template_tone": template_tone,
                "no_progress": no_progress,
                "style_keyword_hits": style_hits,
                "style_penalty": style_pen,
                "reason": str(obj.get("reason", ""))[:120],
            }

        parsed = _extract_first_float(text)
        if parsed is None:
            return {
                "score": 0.5,
                "status": "parse_failed",
                "raw": text,
                "stage": "start" if force_start else "unknown",
                "stage_fit_score": 0.5,
                "objective_score": 0.5,
                "professional_score": 0.5,
                "safety_score": 0.5,
                "benchmark_alignment_score": 0.5,
                "style_penalty": 0.0,
            }

        return {
            "score": _clip(parsed),
            "status": "fallback_float",
            "raw": text,
            "stage": "start" if force_start else "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }
    except aiohttp.ClientResponseError as exc:
        msg = (exc.message or "").strip().replace("\n", " ")
        if len(msg) > 120:
            msg = msg[:120]
        return {
            "score": 0.5,
            "status": f"error:ClientResponseError:{exc.status}:{msg}",
            "raw": "",
            "stage": "start" if force_start else "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }
    except asyncio.TimeoutError:
        return {
            "score": 0.5,
            "status": "error:Timeout",
            "raw": "",
            "stage": "start" if force_start else "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }
    except aiohttp.ClientConnectionError as exc:
        return {
            "score": 0.5,
            "status": f"error:ClientConnectionError:{type(exc).__name__}",
            "raw": "",
            "stage": "start" if force_start else "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "status": f"error:{type(exc).__name__}",
            "raw": "",
            "stage": "start" if force_start else "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
            "benchmark_alignment_score": 0.5,
            "style_penalty": 0.0,
        }
