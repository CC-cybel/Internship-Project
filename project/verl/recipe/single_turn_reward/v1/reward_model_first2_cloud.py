from __future__ import annotations

import json
import re
from typing import Any

import aiohttp


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


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


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
            resp.raise_for_status()
            return await resp.json()


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
        }

    history_text = str(kwargs.get("history_text", "")).strip()
    turn_round = int(_to_float(kwargs.get("turn_round"), 0))
    contact_round = int(_to_float(kwargs.get("contact_round"), 0))
    expected_stage_hint = str(kwargs.get("expected_stage_hint", "")).strip().lower()

    if expected_stage_hint not in {"start", "mid", "contact"}:
        expected_stage_hint = "unknown"

    history_block = history_text if history_text else "<无历史轮次>"

    prompt = (
        "你是医疗咨询转化场景的严格评审员，这个场景是线上和患者对话，以获取患者的联系方式，主要核心是进行联系方式的获取，而非问诊，所有话术均服务于留联目的。\n"
        "你的任务是基于历史对话和当前问题对用户的回答进行评分，评分对象只能是“当前回复”（Action + Content），不是历史回复本身。\n"
        "请先判定阶段，再按阶段细则给分。\n"
        "阶段仅允许: start(开始问诊), mid(中间建立价值), contact(留联收口)。\n"
        "必须依据提供的对话证据评分，禁止臆测。\n"
        "输出必须是 JSON 对象，禁止输出 JSON 之外文本。\n"
        "JSON schema:\n"
        "{\n"
        '  "stage": "start|mid|contact",\n'
        '  "scores": {"stage_fit": 0-1, "objective": 0-1, "professional": 0-1, "safety": 0-1},\n'
        '  "flags": {"premature_contact": true/false, "hard_sell": true/false, "redline": true/false, "invalid_inquiry": true/false, "full_solution_too_early": true/false},\n'
        '  "reason": "<=60字简短中文"\n'
        "}\n"
        "全局维度定义:\n"
        "1) stage_fit: 当前回复与判定阶段是否一致。\n"
        "2) objective: 当前阶段目标达成度。\n"
        "3) professional: 专业性与沟通质量。\n"
        "4) safety: 风险与合规。\n"
        "\n"
        "阶段细则(参考优秀评估规则):\n"
        "A. start 阶段(开始问诊)\n"
        "- 高分行为: 主诉优先, 对症追问, 由浅入深(主诉->细节->影响), 每轮都要推进下一步。\n"
        "- 扣分行为: 无效问诊(离题/查户口/重复问已给信息), 空泛问诊(泛泛一句无诊断价值), 只回答不推进。\n"
        "- 关键要求: 必须有推进动作(明确下一问题/下一步检查点/过渡到价值建立)。\n"
        "- 红线: 对抗、威胁、情绪施压。\n"
        "\n"
        "B. mid 阶段(价值建立)\n"
        "- 高分行为: 给希望+给方向(解释原因/风险趋势/可改善路径), 并自然过渡到留联必要性。\n"
        "- 扣分行为: 直接给完整终局解决方案(让用户无需留联), 机械复读、泛泛安慰、只解释不推进。\n"
        "- 关键要求: mid 的目标是'建立必要性并推进留联'，不是在当前轮把问题彻底解决。\n"
        "- 红线: 辩论压制用户、冷漠忽视痛苦、虚假承诺。\n"
        "\n"
        "C. contact 阶段(留联收口)\n"
        "- 高分行为: 时机自然(价值前置或共情后), 给出明确价值交换(方案/资料/评估), 可使用礼貌的时机压力。\n"
        "- 礼貌时机压力(可加分): 例如提醒'延误可能增加处理成本/错过更佳干预窗口'，语气克制且基于已知信息。\n"
        "- 扣分行为: 指令式索联、乞求式索联、无价值硬要、用户拒绝后无转圜动作。\n"
        "- 红线: 恐吓夸大、绝对化威胁、强制干扰、对拒绝进行质问压迫。\n"
        "\n"
        "评分刻度建议:\n"
        "- 0.90-1.00: 与阶段高度一致, 目标完成明显, 专业且安全。\n"
        "- 0.70-0.89: 整体较好, 有轻微不足。\n"
        "- 0.40-0.69: 阶段或目标部分偏离。\n"
        "- 0.00-0.39: 严重偏离或存在明显风险。\n"
        "- 若 redline=true, safety 必须 <=0.2。\n"
        "\n"
        "对话上下文:\n"
        f"历史轮次:\n{history_block}\n\n"
        f"当前轮次索引(若提供): {turn_round}\n"
        f"参考留联轮次(若提供): {contact_round}\n"
        f"规则侧阶段提示(可参考不可盲从): {expected_stage_hint}\n\n"
        "当前用户问题:\n"
        f"{question}\n\n"
        "待评估助手回复:\n"
        f"{answer}\n"
    )
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }
    try:
        resp = await _chat(api_base=api_base, api_key=api_key, payload=payload, timeout_s=timeout_s)
        text = (
            resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        obj = _extract_json_dict(text)

        if obj is not None:
            scores = obj.get("scores", {}) if isinstance(obj.get("scores"), dict) else {}
            stage_fit = _clip(_to_float(scores.get("stage_fit"), 0.5))
            objective = _clip(_to_float(scores.get("objective"), 0.5))
            professional = _clip(_to_float(scores.get("professional"), 0.5))
            safety = _clip(_to_float(scores.get("safety"), 0.5))

            flags = obj.get("flags", {}) if isinstance(obj.get("flags"), dict) else {}
            redline = bool(flags.get("redline", False))
            invalid_inquiry = bool(flags.get("invalid_inquiry", False))
            full_solution_too_early = bool(flags.get("full_solution_too_early", False))

            sw = _to_float(kwargs.get("stage_weight"), 0.35)
            ow = _to_float(kwargs.get("objective_weight"), 0.35)
            pw = _to_float(kwargs.get("professional_weight"), 0.20)
            fw = _to_float(kwargs.get("safety_weight"), 0.10)
            wsum = max(1e-6, sw + ow + pw + fw)

            score = _clip((sw * stage_fit + ow * objective + pw * professional + fw * safety) / wsum)
            if invalid_inquiry:
                score = _clip(score - 0.12)
            if full_solution_too_early:
                score = _clip(score - 0.10)
            if redline:
                score = min(score, 0.2)

            stage = str(obj.get("stage", "unknown")).strip().lower()
            if stage not in {"start", "mid", "contact"}:
                stage = "unknown"

            return {
                "score": score,
                "status": "ok",
                "raw": text,
                "stage": stage,
                "stage_fit_score": stage_fit,
                "objective_score": objective,
                "professional_score": professional,
                "safety_score": safety,
                "premature_contact": bool(flags.get("premature_contact", False)),
                "hard_sell": bool(flags.get("hard_sell", False)),
                "redline": redline,
                "invalid_inquiry": invalid_inquiry,
                "full_solution_too_early": full_solution_too_early,
                "reason": str(obj.get("reason", ""))[:120],
            }

        parsed = _extract_first_float(text)
        if parsed is None:
            return {
                "score": 0.5,
                "status": "parse_failed",
                "raw": text,
                "stage": "unknown",
                "stage_fit_score": 0.5,
                "objective_score": 0.5,
                "professional_score": 0.5,
                "safety_score": 0.5,
            }
        return {
            "score": _clip(parsed),
            "status": "fallback_float",
            "raw": text,
            "stage": "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
        }
    except Exception as exc:
        return {
            "score": 0.5,
            "status": f"error:{type(exc).__name__}",
            "raw": "",
            "stage": "unknown",
            "stage_fit_score": 0.5,
            "objective_score": 0.5,
            "professional_score": 0.5,
            "safety_score": 0.5,
        }