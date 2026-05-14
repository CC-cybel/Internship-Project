"""Lead reward implementation with rule-based batch evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]


DEFAULT_RULES_FILE = os.getenv(
    "LEAD_REWARD_RULES_FILE", str(Path(__file__).with_name("talk_eval_rule.json"))
)
DEFAULT_MODEL_NAME = os.getenv("LEAD_JUDGE_MODEL") or os.getenv("TALK_REWARD_MODEL", "gpt-5.2")

_CACHE_LOCK = Lock()
_RULES_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_TRACE_LOCK = Lock()


def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _resolve_nonnegative_int(*values: Any) -> int | None:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _squash_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate_text(text: str, max_len: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip(" ，,;；:：") + "..."


def _strip_judge_artifacts(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("<sep>", "，")
    cleaned = re.sub(r"\bEvidence\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAnalysis\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return _squash_whitespace(cleaned)


def _sanitize_violation_evidence(text: str) -> str:
    cleaned = _strip_judge_artifacts(text)
    if not cleaned:
        return ""
    cleaned = re.split(r"[。！？!?；;]\s*|\s*,\s*|\s*，\s*", cleaned, maxsplit=1)[0].strip()
    return _truncate_text(cleaned, 30)


def _sanitize_violation_analysis(text: str) -> str:
    cleaned = _strip_judge_artifacts(text)
    if not cleaned:
        return ""

    if "Analysis:" in text:
        cleaned = _strip_judge_artifacts(text.split("Analysis:", 1)[1])

    meta_markers = [
        "更准确匹配",
        "让我再检查",
        "让我们重新审视",
        "让我们再看",
        "仔细看规则",
        "再看规则",
        "根据严格匹配",
        "最明显的违规其实",
        "更严重的违规",
        "不过，",
        "不过,",
        "然而，",
        "然而,",
        "等等，",
        "等等,",
        "属于被动获联",
        "最明显的违规",
        "最明显的硬伤",
        "让我们看",
        "让我再",
        "let me",
        "however",
        "reconsider",
    ]
    lowered = cleaned.lower()
    cut_idx = -1
    for marker in meta_markers:
        idx = lowered.find(marker.lower())
        if idx > 0:
            cut_idx = idx if cut_idx < 0 else min(cut_idx, idx)
    for marker in [r"规则\s*\d+", r"rule\s*\d+"]:
        match = re.search(marker, cleaned, flags=re.IGNORECASE)
        if match and match.start() > 0:
            cut_idx = match.start() if cut_idx < 0 else min(cut_idx, match.start())
    if cut_idx > 0:
        cleaned = cleaned[:cut_idx].strip()

    first_sentence_parts = re.split(r"[。！？!?]\s*|\n+", cleaned, maxsplit=1)
    if first_sentence_parts:
        cleaned = first_sentence_parts[0].strip()

    cleaned = re.sub(r"^(在尚未[^，。]*前，?)", "", cleaned).strip()
    cleaned = re.sub(r"^(当前回复)", "", cleaned).strip(" ，,")
    comma_parts = [part.strip() for part in re.split(r"[，,]", cleaned) if part.strip()]
    if len(cleaned) > 50 and len(comma_parts) >= 2:
        cleaned = "，".join(comma_parts[:2])

    if not cleaned:
        return ""
    return _truncate_text(cleaned, 55)


def _append_genrm_trace(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with _TRACE_LOCK:
            fd = os.open(abs_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
    except Exception:
        pass


def _load_rules(rules_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(rules_file):
        raise FileNotFoundError(f"Rules file not found: {rules_file}")
    with open(rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)
    if not isinstance(rules, list):
        raise ValueError("Rules file must be a JSON array.")
    return rules


def _get_rules(rules_file: str) -> List[Dict[str, Any]]:
    abs_path = os.path.abspath(rules_file)
    with _CACHE_LOCK:
        if abs_path not in _RULES_CACHE:
            _RULES_CACHE[abs_path] = _load_rules(abs_path)
        return _RULES_CACHE[abs_path]


def _prepare_runtime_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runtime_rules: List[Dict[str, Any]] = []
    for rule in rules:
        runtime_rule = dict(rule)
        runtime_rule["runtime_enabled"] = bool(rule.get("enabled", True))
        runtime_rule["runtime_weight"] = _to_float(rule.get("weight", 1.0), 1.0)
        runtime_rule["runtime_score"] = _to_float(rule.get("score", 0.0), 0.0)
        runtime_rules.append(runtime_rule)
    return runtime_rules


def _parse_action_content(text: str) -> str:
    match = re.match(r"^\s*(?:\[[^\]]+\]|\([^)]+\))\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _extract_final_text(text: str, strict_for_meta: bool = False) -> str:
    final_match = re.search(r"BEGIN_FINAL\s*(.*?)\s*END_FINAL", text, re.DOTALL)
    if final_match:
        return final_match.group(1).strip()
    if strict_for_meta and "BEGIN_META" in text:
        return ""
    return _parse_action_content(text)


def _extract_action_from_meta(text: str) -> str:
    meta_match = re.search(r"BEGIN_META\s*(.*?)\s*END_META", text, re.DOTALL)
    if not meta_match:
        return ""
    meta_block = meta_match.group(1)
    action_match = re.search(r"^[ \t]*action[ \t]*=[ \t]*(.*)$", meta_block, re.MULTILINE)
    return action_match.group(1).strip() if action_match else ""


def _parse_candidate(candidate_text: str) -> Tuple[str, str]:
    action = _extract_action_from_meta(candidate_text)
    content = _extract_final_text(candidate_text, strict_for_meta=True)
    if action or "BEGIN_FINAL" in candidate_text:
        return action, content

    match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)", candidate_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", candidate_text.strip()


def _normalize_turn(turn: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(turn, dict):
        return None

    role = turn.get("role")
    content = turn.get("content")
    if role is None or content is None:
        role = turn.get("from")
        content = turn.get("value")

    if role is None or content is None:
        return None

    role = str(role).strip()
    clean_content = _extract_final_text(str(content), strict_for_meta=True)
    if not clean_content:
        return None
    return role, clean_content


def _build_history(extra_info: Optional[Dict[str, Any]]) -> str:
    if not isinstance(extra_info, dict):
        return ""

    turns: List[Dict[str, Any]] = []
    source = ""
    if isinstance(extra_info.get("dialogue"), list):
        turns = extra_info["dialogue"]
        source = "dialogue"
    elif isinstance(extra_info.get("conversations"), list):
        turns = extra_info["conversations"]
        source = "conversations"
    elif isinstance(extra_info.get("history"), list):
        turns = extra_info["history"]
        source = "history"

    normalized: List[Tuple[str, str]] = []
    for turn in turns:
        parsed = _normalize_turn(turn)
        if parsed:
            normalized.append(parsed)

    exclude_last = extra_info.get("exclude_last_turn")
    if exclude_last is None:
        exclude_last = source == "dialogue"
    if exclude_last and normalized:
        normalized = normalized[:-1]

    return "".join(f"{role}: {content}\n" for role, content in normalized)


def _group_rules_by_dimension(rules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        dim = str(rule.get("rule_dimension", "unknown"))
        grouped.setdefault(dim, []).append(rule)
    return grouped


def _construct_dimension_batch_prompt(
    history_str: str,
    action: str,
    content: str,
    dimension: str,
    rules: List[Dict[str, Any]],
) -> str:
    rule_blocks = []
    for rule in rules:
        rid = rule.get("rule_id")
        rname = rule.get("rule_name_cn", rule.get("rule_name", ""))
        trigger_type = rule.get("trigger", "")
        trigger_cond = rule.get("trigger_condition", "")
        desc = rule.get("description", "")
        special = "；特别注意：该规则需要重点检查Action与Content是否一致" if rid == 11 else ""
        rule_blocks.append(
            f"- rule_id: {rid}\n"
            f"  rule_name: {rname}\n"
            f"  trigger_type: {trigger_type}\n"
            f"  trigger_condition: {trigger_cond}\n"
            f"  description: {desc}{special}"
        )

    rule_text = "\n".join(rule_blocks)
    all_rule_ids = [r.get("rule_id") for r in rules]
    return f"""你是一名专业的医生助理质检（QA）专家。请对同一维度的规则进行批量评估。

评估目标维度：{dimension}

规则列表（必须逐条输出）：
{rule_text}

输入数据：
[对话历史]：
{history_str}

[当前回复-Action]：
{action}

[当前回复-Content]：
{content}

任务要求：
1. 评分对象只能是“当前回复”（Action + Content），不是历史回复本身。
2. 对话历史仅用于理解上下文、判断触发条件，不可把历史里的问题直接算到当前回复上。
3. 只有当当前回复本轮出现违规行为时，才可判定该规则未通过。
4. 对每条规则先判断是否触发（is_triggered）；若触发，再判断是否通过（is_passed）。
5. 若未触发，is_passed 必须为 true。
6. 若是需要上下文的规则（如矛盾、重复问诊、已知信息又问），也必须以“当前回复”是否构成违规为准。
7. 触发与合规分析必须具体，不可只写“符合/不符合”“已触发/未触发”等空泛结论。
8. trigger_analysis 必须包含三部分：A) 触发条件复述；B) 当前回复证据（引用关键词或短语）；C) 触发/不触发结论。
9. compliance_analysis 必须包含三部分：A) 规则要求复述；B) 当前回复中的具体行为证据；C) 通过/违规结论与边界解释。
10. analysis 必须给出完整判定链路：`规则条件 -> 当前回复证据 -> 最终结论`，建议 40 字以上。
11. 若引用上下文，必须说明“上下文如何影响当前回复判定”，但不得把历史行为直接判为当前轮违规。
12. 返回中必须包含本批次所有规则ID：{all_rule_ids}
13. 不要输出任何Markdown或额外文本，只输出JSON。

输出格式（严格）：
{{
  "results": [
    {{
      "rule_id": 1,
      "is_triggered": true,
      "is_passed": false,
      "trigger_analysis": "触发条件：...；当前回复证据：...；触发结论：...",
      "compliance_analysis": "规则要求：...；当前回复行为证据：...；结论：违规/通过，边界说明：...",
      "analysis": "规则条件 -> 当前回复证据 -> 最终结论：..."
    }}
  ]
}}
"""


def _construct_single_call_prompt(
    history_str: str,
    action: str,
    content: str,
    rules: List[Dict[str, Any]],
    protocol: str,
) -> str:
    rule_blocks = []
    for rule in rules:
        rid = rule.get("rule_id")
        rname = rule.get("rule_name_cn", rule.get("rule_name", ""))
        dim = rule.get("rule_dimension", "unknown")
        trigger_type = rule.get("trigger", "")
        trigger_cond = rule.get("trigger_condition", "")
        desc = rule.get("description", "")
        special = "；特别注意：该规则需要重点检查Action与Content是否一致" if rid == 11 else ""
        rule_blocks.append(
            f"- rule_id: {rid}\n"
            f"  rule_dimension: {dim}\n"
            f"  rule_name: {rname}\n"
            f"  trigger_type: {trigger_type}\n"
            f"  trigger_condition: {trigger_cond}\n"
            f"  description: {desc}{special}"
        )

    rules_text = "\n".join(rule_blocks)
    all_rule_ids = [r.get("rule_id") for r in rules]
    if protocol == "violations_only":
        return f"""你是一名专业的医生助理质检（QA）专家。请一次性完整检查所有规则，但只输出确认违规的规则。

规则列表（必须全部检查）：
{rules_text}

输入数据：
[对话历史]：
{history_str}

[当前回复-Action]：
{action}

[当前回复-Content]：
{content}

任务要求：
1. 评分对象只能是“当前回复”（Action + Content），不是历史回复本身。
2. 对话历史仅用于理解上下文、判断触发条件，不可把历史里的问题直接算到当前回复上。
3. 你必须先完整检查全部 rule_id：{all_rule_ids}，再决定哪些规则确认违规。
4. 只有在你确认“当前回复违反了该规则”时，才把该规则写入 violations。
5. 没写入 violations 的规则，表示你已检查过并确认“不违规”（无论是不触发，还是触发但通过）。
6. covered_rule_ids 必须完整列出你已经逐条检查过的全部 rule_id，且必须覆盖 {all_rule_ids}。
7. 每条 violations 都必须提供：
   - rule_id
   - evidence：引用当前回复中的关键短语或概括关键行为，最多 30 个字，禁止长段摘抄
   - analysis：只允许 1 句话，只解释“为什么违反当前 rule_id”，建议 20-40 个字，禁止超过 60 个字
8. analysis 禁止：
   - 提及其他规则或比较哪个规则“更准确”
   - 输出思考过程、自我修正、自我辩论
   - 使用“更准确匹配”“让我再检查”“重新审视”“不过”“然而”“等等”这类元推理表述
9. 如果没有任何违规，也必须输出空数组 `"violations": []`，并保留完整 covered_rule_ids。
10. 不要输出 Markdown、不要输出额外解释，只输出 JSON。

输出格式（严格）：
{{
  "covered_rule_ids": {all_rule_ids},
  "violations": [
    {{
      "rule_id": 25,
      "evidence": "当前回复直接给出可执行缓解建议",
      "analysis": "未留资前直接提供缓解方案，属于违规。"
    }}
  ]
}}
"""

    return f"""你是一名专业的医生助理质检（QA）专家。请一次性评估所有规则。

规则列表（必须逐条输出）：
{rules_text}

输入数据：
[对话历史]：
{history_str}

[当前回复-Action]：
{action}

[当前回复-Content]：
{content}

任务要求：
1. 评分对象只能是“当前回复”（Action + Content），不是历史回复本身。
2. 对话历史仅用于理解上下文、判断触发条件，不可把历史里的问题直接算到当前回复上。
3. 对每条规则先判断是否触发（is_triggered）；若触发，再判断是否通过（is_passed）。
4. 若未触发，is_passed 必须为 true。
5. 请务必覆盖全部 rule_id：{all_rule_ids}
6. analysis 请给出简短但具体的判定理由（建议 15 字以上）。
7. 不要输出Markdown或额外解释，只输出 JSON。

输出格式（严格）：
{{
  "results": [
    {{
      "rule_id": 1,
      "is_triggered": true,
      "is_passed": false,
      "analysis": "规则条件 -> 当前回复证据 -> 最终结论"
    }}
  ]
}}
"""


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "".join(chunks).strip()
    return str(content).strip()


def _extract_chat_response_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return _message_content_to_text(message.get("content", ""))
            if "text" in first:
                return _message_content_to_text(first.get("text", ""))
    for key in ("output_text", "text", "content"):
        if key in response:
            return _message_content_to_text(response.get(key, ""))
    return json.dumps(response, ensure_ascii=False)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_result_items(result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_results = result_data.get("results")
    normalized: List[Dict[str, Any]] = []
    if isinstance(raw_results, list):
        normalized = [x for x in raw_results if isinstance(x, dict)]
    elif isinstance(raw_results, dict):
        for key, value in raw_results.items():
            if isinstance(value, dict):
                row = dict(value)
                if "rule_id" not in row:
                    row["rule_id"] = key
                normalized.append(row)
    return normalized


def _normalize_covered_rule_ids(result_data: Dict[str, Any]) -> List[int]:
    raw = result_data.get("covered_rule_ids")
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_violation_items(result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = result_data.get("violations")
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _build_error_result_map(expected_ids: set[int], error_text: str) -> Dict[int, Dict[str, Any]]:
    return {
        rid: {"is_triggered": False, "is_passed": True, "analysis": error_text, "error": error_text}
        for rid in expected_ids
    }


def _build_single_call_result_map(
    *,
    result_data: Dict[str, Any],
    expected_ids: set[int],
    protocol: str,
) -> Dict[int, Dict[str, Any]]:
    if protocol == "violations_only":
        covered_ids = set(_normalize_covered_rule_ids(result_data))
        if covered_ids != expected_ids:
            missing = sorted(expected_ids - covered_ids)
            extra = sorted(covered_ids - expected_ids)
            raise ValueError(
                "Single-call covered_rule_ids mismatch. "
                f"missing={missing}, extra={extra}"
            )

        result_map: Dict[int, Dict[str, Any]] = {
            rid: {
                "is_triggered": False,
                "is_passed": True,
                "analysis": "已完整检查该规则，未确认违规。",
                "error": None,
            }
            for rid in expected_ids
        }
        for item in _normalize_violation_items(result_data):
            try:
                rid = int(item.get("rule_id"))
            except (TypeError, ValueError):
                continue
            if rid not in expected_ids:
                continue
            evidence = _sanitize_violation_evidence(str(item.get("evidence", "")))
            analysis = _sanitize_violation_analysis(str(item.get("analysis", "")))
            merged = analysis or "模型确认该规则违规。"
            if evidence:
                merged = f"Evidence: {evidence}\nAnalysis: {merged}"
            result_map[rid] = {
                "is_triggered": True,
                "is_passed": False,
                "analysis": merged,
                "error": None,
            }
        return result_map

    normalized_items = _normalize_result_items(result_data)
    result_map: Dict[int, Dict[str, Any]] = {}
    for item in normalized_items:
        try:
            rid = int(item.get("rule_id"))
        except (TypeError, ValueError):
            continue
        if rid not in expected_ids:
            continue

        is_triggered = bool(item.get("is_triggered", False))
        is_passed = bool(item.get("is_passed", True))
        if not is_triggered:
            is_passed = True

        trigger_analysis = str(item.get("trigger_analysis", "")).strip()
        compliance_analysis = str(item.get("compliance_analysis", "")).strip()
        analysis = str(item.get("analysis", "")).strip()
        if not analysis:
            if is_triggered:
                analysis = (
                    f"Trigger Analysis: {trigger_analysis}\n"
                    f"Compliance Analysis: {compliance_analysis}"
                ).strip()
            else:
                analysis = trigger_analysis or "规则未触发。"

        result_map[rid] = {
            "is_triggered": is_triggered,
            "is_passed": is_passed,
            "trigger_analysis": trigger_analysis,
            "compliance_analysis": compliance_analysis,
            "analysis": analysis,
            "error": None,
        }

    for rid in expected_ids:
        if rid not in result_map:
            err = f"Missing rule_id={rid} in single-call output."
            result_map[rid] = {
                "is_triggered": False,
                "is_passed": True,
                "analysis": err,
                "error": err,
            }
    return result_map


async def _chat_complete(
    reward_router_address: str,
    payload: Dict[str, Any],
    timeout_s: float = 60.0,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> Dict[str, Any]:
    if aiohttp is None:
        raise ImportError("aiohttp package is required. Please run: pip install aiohttp")
    base = str(reward_router_address).strip()
    if base.startswith("http://") or base.startswith("https://"):
        base_url = base.rstrip("/")
    else:
        base_url = f"http://{base.rstrip('/')}"
    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    total_attempts = max(1, int(max_retries))
    backoff_s = max(0.0, float(retry_backoff_s))
    last_exc: Exception | None = None
    for attempt in range(total_attempts):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < total_attempts - 1:
                await asyncio.sleep(backoff_s * (2**attempt))
    assert last_exc is not None
    raise last_exc


async def _evaluate_rule_group(
    reward_router_address: str,
    judge_model: str,
    history_str: str,
    dimension: str,
    rules: List[Dict[str, Any]],
    action: str,
    content: str,
    timeout_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool,
    thinking_budget: int | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    trace_enabled: bool = False,
    trace_path: str = "",
    trace_common: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    expected_ids: set[int] = set()
    for r in rules:
        rid = r.get("rule_id")
        if rid is None:
            continue
        try:
            expected_ids.add(int(rid))
        except (TypeError, ValueError):
            continue
    if not expected_ids:
        return {}

    payload: Dict[str, Any] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "You are a strict and professional QA evaluator."},
            {
                "role": "user",
                "content": _construct_dimension_batch_prompt(
                    history_str=history_str,
                    action=action,
                    content=content,
                    dimension=dimension,
                    rules=rules,
                ),
            },
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        payload["extra_body"] = {"enable_thinking": True}
    if thinking_budget is not None:
        payload["thinking_budget"] = int(thinking_budget)
    payload_messages = payload.get("messages")
    if isinstance(payload_messages, list) and len(payload_messages) > 1 and isinstance(payload_messages[1], dict):
        judge_prompt = str(payload_messages[1].get("content", ""))
    else:
        judge_prompt = ""

    try:
        resp = await _chat_complete(
            reward_router_address=reward_router_address,
            payload=payload,
            timeout_s=timeout_s,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        result_text = _extract_chat_response_text(resp)
        if trace_enabled:
            _append_genrm_trace(
                trace_path,
                {
                    "ts": time.time(),
                    "event": "genrm_io",
                    "rm_name": "lead_reward",
                    "stage": "rule_group_eval",
                    "dimension": dimension,
                    "rule_ids": sorted(expected_ids),
                    "judge_model": judge_model,
                    "request_prompt": judge_prompt,
                    "response_text": result_text,
                    "error": None,
                    "meta": trace_common or {},
                },
            )
        lower_text = result_text.lstrip().lower()
        if lower_text.startswith("<!doctype html") or lower_text.startswith("<html"):
            raise ValueError(
                "Received HTML instead of JSON from model API. "
                "Check reward_router_address/model routing."
            )
        result_data = _extract_json_object(result_text)
        if not isinstance(result_data, dict):
            raise ValueError(f"Batch JSON parsing failed. Raw output: {result_text}")

        raw_results = result_data.get("results")
        normalized_items: List[Dict[str, Any]] = []
        if isinstance(raw_results, list):
            normalized_items = [x for x in raw_results if isinstance(x, dict)]
        elif isinstance(raw_results, dict):
            for key, value in raw_results.items():
                if isinstance(value, dict):
                    item = dict(value)
                    if "rule_id" not in item:
                        item["rule_id"] = key
                    normalized_items.append(item)

        group_results: Dict[int, Dict[str, Any]] = {}
        for item in normalized_items:
            try:
                rid = int(item.get("rule_id"))
            except (TypeError, ValueError):
                continue
            if rid not in expected_ids:
                continue

            is_triggered = bool(item.get("is_triggered", False))
            is_passed = bool(item.get("is_passed", True))
            if not is_triggered:
                is_passed = True

            trigger_analysis = str(item.get("trigger_analysis", "")).strip()
            compliance_analysis = str(item.get("compliance_analysis", "")).strip()
            analysis = str(item.get("analysis", "")).strip()
            if not analysis:
                if is_triggered:
                    analysis = (
                        f"Trigger Analysis: {trigger_analysis}\n"
                        f"Compliance Analysis: {compliance_analysis}"
                    ).strip()
                else:
                    analysis = trigger_analysis or "规则未触发。"

            group_results[rid] = {
                "is_triggered": is_triggered,
                "is_passed": is_passed,
                "trigger_analysis": trigger_analysis,
                "compliance_analysis": compliance_analysis,
                "analysis": analysis,
                "error": None,
            }

        for rid in expected_ids:
            if rid not in group_results:
                group_results[rid] = {
                    "is_triggered": False,
                    "is_passed": True,
                    "analysis": f"Missing rule_id={rid} in batch output for dimension={dimension}.",
                    "error": f"Missing rule_id={rid} in batch output for dimension={dimension}.",
                }
        return group_results
    except Exception as exc:
        err = f"Batch evaluation failed for dimension={dimension}: {exc}"
        if trace_enabled:
            _append_genrm_trace(
                trace_path,
                {
                    "ts": time.time(),
                    "event": "genrm_io",
                    "rm_name": "lead_reward",
                    "stage": "rule_group_eval",
                    "dimension": dimension,
                    "rule_ids": sorted(expected_ids),
                    "judge_model": judge_model,
                    "request_prompt": judge_prompt,
                    "response_text": "",
                    "error": err,
                    "meta": trace_common or {},
                },
            )
        return {
            rid: {"is_triggered": False, "is_passed": True, "analysis": err, "error": err}
            for rid in expected_ids
        }


async def _evaluate_rules_by_dimension(
    *,
    enabled_rules: List[Dict[str, Any]],
    reward_router_address: str,
    judge_model: str,
    history_str: str,
    action: str,
    content: str,
    timeout_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool,
    thinking_budget: int | None = None,
    dimension_workers: int = 5,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    trace_enabled: bool = False,
    trace_path: str = "",
    trace_common: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    grouped_rules = _group_rules_by_dimension(enabled_rules)
    batched_results: Dict[int, Dict[str, Any]] = {}
    dim_items = list(grouped_rules.items())
    worker_count = max(1, min(dimension_workers, len(dim_items) if dim_items else 1))
    semaphore = asyncio.Semaphore(worker_count)

    async def _run_dimension(dimension: str, group: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        async with semaphore:
            return await _evaluate_rule_group(
                reward_router_address=reward_router_address,
                judge_model=judge_model,
                history_str=history_str,
                dimension=dimension,
                rules=group,
                action=action,
                content=content,
                timeout_s=timeout_s,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                api_key=api_key,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                trace_enabled=trace_enabled,
                trace_path=trace_path,
                trace_common=trace_common,
            )

    if dim_items:
        results = await asyncio.gather(
            *[_run_dimension(dimension, group) for dimension, group in dim_items],
            return_exceptions=True,
        )
        for idx, item in enumerate(results):
            dimension, group = dim_items[idx]
            if isinstance(item, Exception):
                err = f"Batch evaluation failed for dimension={dimension}: {item}"
                expected_ids: set[int] = set()
                for rule in group:
                    rid = rule.get("rule_id")
                    if rid is None:
                        continue
                    try:
                        expected_ids.add(int(rid))
                    except (TypeError, ValueError):
                        continue
                batched_results.update(_build_error_result_map(expected_ids, err))
            else:
                batched_results.update(item)
    return batched_results


async def _evaluate_all_rules_single_call(
    *,
    enabled_rules: List[Dict[str, Any]],
    reward_router_address: str,
    judge_model: str,
    history_str: str,
    action: str,
    content: str,
    timeout_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool,
    thinking_budget: int | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    trace_enabled: bool = False,
    trace_path: str = "",
    trace_common: Optional[Dict[str, Any]] = None,
    protocol: str = "violations_only",
) -> Dict[int, Dict[str, Any]]:
    expected_ids: set[int] = set()
    for rule in enabled_rules:
        rid = rule.get("rule_id")
        if rid is None:
            continue
        try:
            expected_ids.add(int(rid))
        except (TypeError, ValueError):
            continue
    if not expected_ids:
        return {}

    payload: Dict[str, Any] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "You are a strict and professional QA evaluator."},
            {
                "role": "user",
                "content": _construct_single_call_prompt(
                    history_str=history_str,
                    action=action,
                    content=content,
                    rules=enabled_rules,
                    protocol=protocol,
                ),
            },
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        payload["extra_body"] = {"enable_thinking": True}
    if thinking_budget is not None:
        payload["thinking_budget"] = int(thinking_budget)
    judge_prompt = str(payload["messages"][1]["content"])

    try:
        resp = await _chat_complete(
            reward_router_address=reward_router_address,
            payload=payload,
            timeout_s=timeout_s,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        result_text = _extract_chat_response_text(resp)
        if trace_enabled:
            _append_genrm_trace(
                trace_path,
                {
                    "ts": time.time(),
                    "event": "genrm_io",
                    "rm_name": "lead_reward",
                    "stage": "single_call_eval",
                    "rule_ids": sorted(expected_ids),
                    "single_call_protocol": protocol,
                    "judge_model": judge_model,
                    "request_prompt": judge_prompt,
                    "response_text": result_text,
                    "error": None,
                    "meta": trace_common or {},
                },
            )
        lower_text = result_text.lstrip().lower()
        if lower_text.startswith("<!doctype html") or lower_text.startswith("<html"):
            raise ValueError(
                "Received HTML instead of JSON from model API. "
                "Check reward_router_address/model routing."
            )
        result_data = _extract_json_object(result_text)
        if not isinstance(result_data, dict):
            raise ValueError(f"Single-call JSON parsing failed. Raw output: {result_text}")
        return _build_single_call_result_map(
            result_data=result_data,
            expected_ids=expected_ids,
            protocol=protocol,
        )
    except Exception as exc:
        err = f"Single-call evaluation failed: {exc}"
        if trace_enabled:
            _append_genrm_trace(
                trace_path,
                {
                    "ts": time.time(),
                    "event": "genrm_io",
                    "rm_name": "lead_reward",
                    "stage": "single_call_eval",
                    "rule_ids": sorted(expected_ids),
                    "single_call_protocol": protocol,
                    "judge_model": judge_model,
                    "request_prompt": judge_prompt,
                    "response_text": "",
                    "error": err,
                    "meta": trace_common or {},
                },
            )
        raise RuntimeError(err) from exc


async def _compute_score_detail(
    answer: str,
    extra_info: Optional[Dict[str, Any]],
    reward_router_address: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    if aiohttp is None:
        raise ImportError("aiohttp package is required. Please run: pip install aiohttp")

    judge_model = (
        kwargs.get("lead_judge_model")
        or kwargs.get("judge_model")
        or kwargs.get("model_name")
        or DEFAULT_MODEL_NAME
    )
    if not judge_model:
        raise ValueError("judge_model_missing")
    rules_file = kwargs.get("lead_rules_file") or kwargs.get("rules_file") or DEFAULT_RULES_FILE
    provided_rules = kwargs.get("lead_rules")
    if provided_rules is None:
        provided_rules = kwargs.get("rules")
    enable_thinking = _to_bool(kwargs.get("enable_thinking", False), default=False)
    thinking_budget = _resolve_nonnegative_int(
        kwargs.get("lead_thinking_budget"),
        kwargs.get("thinking_budget"),
    )
    max_retries = max(
        1,
        int(
            _to_float(
                kwargs.get("lead_judge_max_retries", kwargs.get("judge_max_retries", 3)),
                3.0,
            )
        ),
    )
    retry_backoff_s = max(
        0.0,
        _to_float(
            kwargs.get(
                "lead_judge_retry_backoff_s",
                kwargs.get("judge_retry_backoff_s", 1.0),
            ),
            1.0,
        ),
    )
    dimension_workers = int(kwargs.get("dimension_workers", 5))
    base_score = float(kwargs.get("base_score", 100.0))
    normalize = _to_bool(kwargs.get("normalize", True), default=True)
    clip_reward = _to_bool(kwargs.get("clip_reward", True), default=True)
    include_rule_details = _to_bool(kwargs.get("include_rule_details", False), default=False)
    timeout_s = float(kwargs.get("judge_timeout_s", 60.0))
    temperature = float(kwargs.get("lead_judge_temperature", 0.01))
    top_p = float(kwargs.get("lead_judge_top_p", 1.0))
    max_tokens = int(kwargs.get("lead_judge_max_tokens", 4096))
    api_key = kwargs.get("lead_api_key") or kwargs.get("api_key") or kwargs.get("llm_api_key")
    trace_enabled = _to_bool(kwargs.get("collect_genrm_io", False), default=False)
    trace_path = str(kwargs.get("genrm_io_path", "/tmp/genrm_io.jsonl"))
    eval_mode = str(kwargs.get("lead_eval_mode", "dimension_batch")).strip().lower()
    if eval_mode in {"batch", "dimension", "dimension_batches"}:
        eval_mode = "dimension_batch"
    if eval_mode not in {"dimension_batch", "single_call"}:
        raise ValueError(f"Unsupported lead_eval_mode={eval_mode!r}")
    single_call_protocol = str(kwargs.get("lead_single_call_protocol", "violations_only")).strip().lower()
    if single_call_protocol not in {"violations_only", "full_results"}:
        raise ValueError(f"Unsupported lead_single_call_protocol={single_call_protocol!r}")
    single_call_fallback = _to_bool(
        kwargs.get(
            "lead_single_call_fallback_to_batch",
            kwargs.get("lead_single_call_fallback_to_dimension_batch", True),
        ),
        default=True,
    )
    sample_id = ""
    if isinstance(extra_info, dict):
        sample_id = str(extra_info.get("sample_id", ""))
    trace_common = {
        "sample_id": sample_id,
        "source": extra_info.get("source") if isinstance(extra_info, dict) else None,
        "conv_id": extra_info.get("conv_id") if isinstance(extra_info, dict) else None,
        "turn_id": extra_info.get("turn_id") if isinstance(extra_info, dict) else None,
    }

    if provided_rules is not None:
        if not isinstance(provided_rules, list):
            raise ValueError("lead_rules must be a list when provided")
        rules = [dict(rule) for rule in provided_rules if isinstance(rule, dict)]
    else:
        rules = _get_rules(rules_file)
    runtime_rules = _prepare_runtime_rules(rules)
    history_str = _build_history(extra_info)
    action, content = _parse_candidate(answer)

    final_score = base_score
    rule_details: List[Dict[str, Any]] = []
    enabled_rules = [r for r in runtime_rules if r.get("runtime_enabled", True)]
    active_rules_count = len(enabled_rules)
    result_map: Dict[int, Dict[str, Any]] = {}
    effective_eval_mode = eval_mode
    if eval_mode == "single_call":
        try:
            result_map = await _evaluate_all_rules_single_call(
                enabled_rules=enabled_rules,
                reward_router_address=reward_router_address,
                judge_model=judge_model,
                history_str=history_str,
                action=action,
                content=content,
                timeout_s=timeout_s,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                api_key=api_key,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                trace_enabled=trace_enabled,
                trace_path=trace_path,
                trace_common=trace_common,
                protocol=single_call_protocol,
            )
        except Exception as exc:
            if not single_call_fallback:
                expected_ids: set[int] = set()
                for rule in enabled_rules:
                    rid = rule.get("rule_id")
                    if rid is None:
                        continue
                    try:
                        expected_ids.add(int(rid))
                    except (TypeError, ValueError):
                        continue
                result_map = _build_error_result_map(expected_ids, str(exc))
            else:
                effective_eval_mode = "dimension_batch_fallback"
                result_map = await _evaluate_rules_by_dimension(
                    enabled_rules=enabled_rules,
                    reward_router_address=reward_router_address,
                    judge_model=judge_model,
                    history_str=history_str,
                    action=action,
                    content=content,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    dimension_workers=dimension_workers,
                    api_key=api_key,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    trace_enabled=trace_enabled,
                    trace_path=trace_path,
                    trace_common=trace_common,
                )
    else:
        result_map = await _evaluate_rules_by_dimension(
            enabled_rules=enabled_rules,
            reward_router_address=reward_router_address,
            judge_model=judge_model,
            history_str=history_str,
            action=action,
            content=content,
            timeout_s=timeout_s,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            dimension_workers=dimension_workers,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            trace_enabled=trace_enabled,
            trace_path=trace_path,
            trace_common=trace_common,
        )

    for rule in runtime_rules:
        if not rule.get("runtime_enabled", True):
            if include_rule_details:
                rule_details.append(
                    {
                        "rule_id": rule.get("rule_id"),
                        "rule_name": rule.get("rule_name"),
                        "rule_name_cn": rule.get("rule_name_cn", rule.get("rule_name")),
                        "original_score": rule.get("score"),
                        "runtime_score": rule.get("runtime_score"),
                        "runtime_weight": rule.get("runtime_weight"),
                        "enabled": False,
                        "is_triggered": False,
                        "is_passed": True,
                        "deduction": 0.0,
                        "analysis": "Rule disabled by rules file.",
                        "error": None,
                    }
                )
            continue

        rid_raw = rule.get("rule_id")
        try:
            rid = int(rid_raw)
        except (TypeError, ValueError):
            if include_rule_details:
                rule_details.append(
                    {
                        "rule_id": rid_raw,
                        "rule_name": rule.get("rule_name"),
                        "rule_name_cn": rule.get("rule_name_cn", rule.get("rule_name")),
                        "original_score": rule.get("score"),
                        "runtime_score": rule.get("runtime_score"),
                        "runtime_weight": rule.get("runtime_weight"),
                        "enabled": True,
                        "is_triggered": False,
                        "is_passed": True,
                        "deduction": 0.0,
                        "analysis": f"Invalid rule_id={rid_raw}.",
                        "error": f"Invalid rule_id={rid_raw}.",
                    }
                )
            continue

        res = result_map.get(
            rid,
            {
                "is_triggered": False,
                "is_passed": True,
                "analysis": f"Missing aggregated result for rule_id={rid}.",
                "error": f"Missing aggregated result for rule_id={rid}.",
            },
        )

        deduction = 0.0
        if res.get("is_triggered") and not res.get("is_passed"):
            deduction = _to_float(rule.get("runtime_score", rule.get("score", 0.0)), 0.0) * _to_float(
                rule.get("runtime_weight", 1.0), 1.0
            )
            final_score += deduction

        if include_rule_details:
            rule_details.append(
                {
                    "rule_id": rule.get("rule_id"),
                    "rule_name": rule.get("rule_name"),
                    "rule_name_cn": rule.get("rule_name_cn", rule.get("rule_name")),
                    "original_score": rule.get("score"),
                    "runtime_score": rule.get("runtime_score"),
                    "runtime_weight": rule.get("runtime_weight"),
                    "enabled": True,
                    "is_triggered": res.get("is_triggered"),
                    "is_passed": res.get("is_passed"),
                    "deduction": deduction,
                    "analysis": res.get("analysis", ""),
                    "error": res.get("error"),
                }
            )

    if normalize and base_score != 0:
        reward = final_score / base_score
        if clip_reward:
            reward = _clip(reward)
    else:
        reward = final_score

    result = {
        "score": float(reward),
        "reward": float(reward),
        "base_score": float(base_score),
        "final_score": float(final_score),
        "rules_count": len(rules),
        "active_rules_count": active_rules_count,
        "eval_mode": eval_mode,
        "effective_eval_mode": effective_eval_mode,
        "single_call_protocol": single_call_protocol if eval_mode == "single_call" else None,
        "status": "ok",
    }
    if include_rule_details:
        result["rule_details"] = rule_details
    if trace_enabled:
        _append_genrm_trace(
            trace_path,
            {
                "ts": time.time(),
                "event": "genrm_summary",
                "rm_name": "lead_reward",
                "status": result.get("status", "ok"),
                "score": result.get("score", 0.0),
                "sample_id": sample_id,
                "source": trace_common.get("source"),
                "conv_id": trace_common.get("conv_id"),
                "turn_id": trace_common.get("turn_id"),
                "active_rules_count": active_rules_count,
                "eval_mode": eval_mode,
                "effective_eval_mode": effective_eval_mode,
                "single_call_protocol": single_call_protocol if eval_mode == "single_call" else None,
                "triggered_rule_count": sum(
                    1 for detail in rule_details if isinstance(detail, dict) and detail.get("is_triggered")
                )
                if include_rule_details
                else None,
            },
        )
    return result


async def compute_lead_score(
    question: str,
    answer: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
    reward_router_address: str | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return {'score': float in [0,1], 'raw': str, 'status': str}."""
    del question
    del ground_truth

    if not reward_router_address:
        return {"score": 0.0, "raw": "", "status": "reward_router_address_missing"}

    try:
        details = await _compute_score_detail(
            answer=answer,
            extra_info=extra_info,
            reward_router_address=reward_router_address,
            **kwargs,
        )
        return {
            "score": float(details.get("score", 0.0)),
            "raw": json.dumps(details, ensure_ascii=False),
            "status": str(details.get("status", "ok")),
        }
    except Exception as exc:
        return {"score": 0.0, "raw": "", "status": f"lead_error:{type(exc).__name__}"}


__all__ = ["compute_lead_score"]
