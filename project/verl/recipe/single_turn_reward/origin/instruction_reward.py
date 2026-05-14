"""Template for instruction-following reward collaborator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import aiohttp

INSTRUCTION_PROMPT_TEMPLATE = ""
DEFAULT_RUBRICS_PATH = str(Path(__file__).with_name("rubrics_instruction_following.json"))
SYSTEM_ROUND_PATTERN = re.compile(r"【系统数据：当前第\s*\d+\s*轮】")
_TRACE_LOCK = Lock()
_RULE_MAP_CACHE_LOCK = Lock()
_RULE_MAP_CACHE: dict[str, list[dict[str, Any]]] = {}


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


def _to_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _clean_context_text(text: str) -> str:
    clean_text = extract_final_response(str(text or ""))
    clean_text = SYSTEM_ROUND_PATTERN.sub("", clean_text).strip()
    return " ".join(clean_text.split())


def _extract_history_list(extra_info: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("history", "conversations", "dialogue"):
        value = extra_info.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_history_turn(turn: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(turn, dict):
        return None

    role = turn.get("from")
    content = turn.get("value")
    if role is None or content is None:
        role = turn.get("role")
        content = turn.get("content")

    if role is None or content is None:
        return None

    role_str = str(role).strip().lower()
    if role_str in {"human", "user"}:
        role_label = "用户"
    elif role_str in {"assistant", "gpt", "bot"}:
        role_label = "助手"
    else:
        return None

    cleaned_content = _clean_context_text(str(content))
    if not cleaned_content:
        return None
    return role_label, cleaned_content


def _extract_system_prompt(extra_info: dict[str, Any]) -> str:
    for key in ("system_prompt", "original_system_prompt", "system", "prompt_system"):
        val = extra_info.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    prompt_messages = extra_info.get("prompt")
    if isinstance(prompt_messages, list):
        for msg in prompt_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role == "system":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _append_genrm_trace(path: str, record: dict[str, Any]) -> None:
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


async def _chat_complete(
    reward_router_address: str,
    payload: dict[str, Any],
    timeout_s: float = 60.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    if reward_router_address.startswith("http://") or reward_router_address.startswith(
        "https://"
    ):
        base_url = reward_router_address.rstrip("/")
    else:
        base_url = f"http://{reward_router_address}"

    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()


def _normalize_json_bools(obj: Any) -> Any:
    """Recursively normalize Python booleans to lowercase strings in JSON."""
    if isinstance(obj, dict):
        return {k: _normalize_json_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_json_bools(item) for item in obj]
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    else:
        return obj


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

    cleaned = re.sub(r"^(当前回复)", "", cleaned).strip(" ，,")
    comma_parts = [part.strip() for part in re.split(r"[，,]", cleaned) if part.strip()]
    if len(cleaned) > 50 and len(comma_parts) >= 2:
        cleaned = "，".join(comma_parts[:2])

    if not cleaned:
        return ""
    return _truncate_text(cleaned, 55)


def _normalize_rule_id_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def load_rubrics(rubrics_path: str) -> list[dict[str, Any]]:
    """Load rubrics configuration from JSON/JSONL file.

    Args:
        rubrics_path: Path to rubrics JSON/JSONL file

    Returns:
        List of rule configurations

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty or invalid
    """
    path = Path(rubrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Rubrics file not found: {rubrics_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            if rubrics_path.endswith(".jsonl"):
                rubrics = []
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if line:
                            rubrics.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"JSON parse error at line {line_num}: {e}")
            else:
                content = json.load(f)
                rubrics = content if isinstance(content, list) else [content]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load rubrics: {e}")

    if not rubrics:
        raise ValueError("Rubrics file is empty")

    return rubrics


class RuleMapper:
    """Rule mapper: dynamically activate rules based on System Prompt.

    Uses LLM to analyze system prompts and select applicable rules
    from the candidate rule list. Results are cached by system prompt hash.
    """

    def __init__(
        self,
        reward_router_address: str,
        judge_model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.1,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        verbose: bool = False,
    ):
        """Initialize RuleMapper.

        Args:
            reward_router_address: Address for reward router API
            judge_model: Model name for LLM calls
            timeout_s: Timeout for API calls
            temperature: Temperature for generation
            api_key: API key for authentication
            verbose: Whether to print debug logs
        """
        self.reward_router_address = reward_router_address
        self.judge_model = judge_model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.api_key = api_key
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.verbose = verbose
        self.cache: dict[str, list[dict[str, Any]]] = {}

    def _build_map_prompt(
        self, system_prompt: str, all_rules: list[dict[str, Any]]
    ) -> str:
        """Build prompt for rule mapping."""
        rule_summary = []
        for r in all_rules:
            rule_summary.append(
                {
                    "rule_id": r["rule_id"],
                    "rule_name": r["rule_name"],
                    "description": r["description"],
                    "config_keys": list(r.get("config", {}).keys()),
                }
            )

        return f"""你是一名规则配置解析专家。请分析以下【Agent System Prompt】，从【候选规则列表】中筛选出适用的规则，并提取具体的参数值（如轮次 N）。

【Agent System Prompt】
{system_prompt}

【候选规则列表】
{json.dumps(rule_summary, ensure_ascii=False)}

【任务】
1. 筛选：只保留 System Prompt 中明确提及或隐含要求的规则。
2. 提取：如果 Prompt 中提到了具体轮次（如"第 5 轮"），请更新规则 config 中的对应参数。
3. 忽略：如果 Prompt 明确说"跳过某项"，则不要包含该规则。

【输出格式】
仅输出 JSON 对象，格式如下：
{{
    "active_rule_ids": [1, 5, 10],
    "config_overrides": {{
        "10": {{ "target_turn": 5 }},
        "17": {{ "deadline_turn": 3 }}
    }}
}}
"""

    async def map_rules(
        self, system_prompt: str, all_rules: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Map rules based on system prompt using LLM.

        Args:
            system_prompt: The system prompt to analyze
            all_rules: All candidate rules

        Returns:
            List of active rules with dynamic configurations
        """
        if not system_prompt or not system_prompt.strip():
            return all_rules

        cache_key_payload = {
            "system_prompt": system_prompt,
            "rule_ids": [r.get("rule_id") for r in all_rules],
            "rule_descriptions": [r.get("description", "") for r in all_rules],
        }
        prompt_hash = hashlib.md5(
            json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if prompt_hash in self.cache:
            return json.loads(json.dumps(self.cache[prompt_hash]))
        with _RULE_MAP_CACHE_LOCK:
            if prompt_hash in _RULE_MAP_CACHE:
                cached = json.loads(json.dumps(_RULE_MAP_CACHE[prompt_hash]))
                self.cache[prompt_hash] = cached
                return cached

        map_prompt = self._build_map_prompt(system_prompt, all_rules)

        payload = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": map_prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await _chat_complete(
                    reward_router_address=self.reward_router_address,
                    payload=payload,
                    timeout_s=self.timeout_s,
                    api_key=self.api_key,
                )
                result_text = (
                    resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                if not result_text:
                    raise ValueError("LLM returned empty content")

                # 使用 extract_json_from_text 处理可能的 markdown 代码块
                cleaned_json = extract_json_from_text(result_text)
                mapping_result = json.loads(cleaned_json)

                active_rules = self._process_mapping_result(mapping_result, all_rules)
                cached_rules = json.loads(json.dumps(active_rules))
                self.cache[prompt_hash] = cached_rules
                with _RULE_MAP_CACHE_LOCK:
                    _RULE_MAP_CACHE[prompt_hash] = json.loads(json.dumps(active_rules))
                return json.loads(json.dumps(cached_rules))
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_s * (2**attempt))

        assert last_exc is not None
        if self.verbose:
            print(f"  ⚠️ 规则映射失败: {type(last_exc).__name__}: {str(last_exc)}")
            print(f"     🔄 Fallback 到全部 {len(all_rules)} 条规则")
        return all_rules

    def _process_mapping_result(
        self, mapping_result: dict[str, Any], all_rules: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process mapping result and generate final active rules list."""
        active_ids = mapping_result.get("active_rule_ids", [])
        overrides = mapping_result.get("config_overrides", {})

        final_rules = []
        for rule in all_rules:
            if rule["rule_id"] in active_ids:
                dynamic_rule = json.loads(json.dumps(rule))
                if str(rule["rule_id"]) in overrides:
                    if "config" not in dynamic_rule:
                        dynamic_rule["config"] = {}
                    dynamic_rule["config"].update(overrides[str(rule["rule_id"])])
                dynamic_rule["_is_dynamic"] = True
                final_rules.append(dynamic_rule)

        return final_rules


def extract_field_value(eval_result: dict[str, Any], field_name: str) -> Any:
    """Extract field value from evaluation result, supporting nested format."""
    value = eval_result.get(field_name)

    if isinstance(value, dict) and "result" in value:
        value = value["result"]

    if value is not None:
        value_str = str(value).lower()
        if value_str in ["none", "null", ""]:
            return None
        if value_str in ["true", "false"]:
            return value_str
        if value_str in ["yes", "no"]:
            return value_str
        if value_str == "partial":
            return "partially"
        return value_str

    return value


def _check_always_trigger(eval_result: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Always type rule check."""
    is_violation = extract_field_value(eval_result, "is_violation")
    if is_violation == "true" or is_violation is True:
        return True

    for field_name, value in eval_result.items():
        val = extract_field_value(eval_result, field_name)
        if field_name.endswith("_detected") and val in ["true", True, "yes"]:
            return True
        if field_name == "phrase_evidence" and val and len(str(val)) > 0:
            if extract_field_value(eval_result, "is_violation") in [
                "true",
                True,
                "yes",
            ]:
                return True

    return False


def _check_turn_based_trigger(
    eval_result: dict[str, Any], rule: dict[str, Any], current_turn: int = 1
) -> bool:
    """Turn-Based type rule check."""
    config = rule.get("config", {})
    deadline_turn = config.get("deadline_turn")
    target_turn = config.get("target_turn")

    if deadline_turn is not None:
        if current_turn > deadline_turn:
            return False

    if target_turn is not None:
        if current_turn != target_turn:
            return False

    is_violation = extract_field_value(eval_result, "is_violation")
    is_rewarded = extract_field_value(eval_result, "is_rewarded")

    if is_violation in ["true", True, "yes"]:
        return True
    if is_rewarded in ["true", True, "yes"]:
        return True

    return False


def _check_context_trigger(
    eval_result: dict[str, Any], rule: dict[str, Any], current_turn: int = 1
) -> bool:
    """Context-Triggered type rule check."""
    rule_name = rule["rule_name"]
    config = rule.get("config", {})
    deadline_turn = config.get("deadline_turn")

    if deadline_turn is not None:
        if current_turn > deadline_turn:
            return False

    if rule_name == "logic_missed_phone_opportunity":
        trust = extract_field_value(eval_result, "trust_level_assessment")
        value = extract_field_value(eval_result, "value_provided")
        phone_asked = extract_field_value(eval_result, "phone_ask_attempted")
        timing = extract_field_value(eval_result, "timing_assessment")

        return (
            trust in ["medium", "high"]
            and value in ["yes", "partially"]
            and phone_asked in ["no", False]
            and timing == "missed"
        )

    if rule_name == "med_forced_symptom":
        vague = extract_field_value(eval_result, "vague_question_detected")
        return vague in ["true", True, "yes"]

    if rule_name == "logic_primary_only":
        multiple = extract_field_value(eval_result, "multiple_diseases_mentioned")
        aligned = extract_field_value(eval_result, "focus_aligned")
        violation = extract_field_value(eval_result, "is_violation")

        return (
            multiple in ["yes", True]
            and aligned in ["no", False]
            and violation in ["true", True, "yes"]
        )

    is_violation = extract_field_value(eval_result, "is_violation")
    return is_violation in ["true", True, "yes"]


def _check_event_trigger(eval_result: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Event-Triggered type rule check."""
    rule_name = rule["rule_name"]

    if rule_name == "conv_medication_phone":
        user_mentioned = extract_field_value(eval_result, "user_mentioned_medication")
        phone_asked = extract_field_value(eval_result, "phone_asked_with_reason")
        violation = extract_field_value(eval_result, "is_violation")

        return (
            user_mentioned in ["yes", True]
            and phone_asked in ["no", False]
            and violation in ["true", True, "yes"]
        )

    if rule_name == "conv_ask_wechat_backup":
        refused = extract_field_value(eval_result, "user_refused_phone")
        wechat_asked = extract_field_value(eval_result, "wechat_asked")
        violation = extract_field_value(eval_result, "is_violation")

        return (
            refused in ["yes", True]
            and wechat_asked in ["no", False]
            and violation in ["true", True, "yes"]
        )

    is_violation = extract_field_value(eval_result, "is_violation")
    return is_violation in ["true", True, "yes"]


def check_rule_triggered(
    eval_result: dict[str, Any],
    rule: dict[str, Any],
    current_turn: int = 1,
    slot_status: Optional[dict[str, int]] = None,
) -> bool:
    """Check if rule is triggered based on LLM evaluation result."""
    trigger_type = rule["trigger"]

    try:
        if trigger_type == "Always":
            return _check_always_trigger(eval_result, rule)

        elif trigger_type == "Turn-Based":
            return _check_turn_based_trigger(eval_result, rule, current_turn)

        elif trigger_type == "Context-Triggered":
            return _check_context_trigger(eval_result, rule, current_turn)

        elif trigger_type == "Event-Triggered":
            return _check_event_trigger(eval_result, rule)

        return False

    except Exception:
        return False


def calculate_sample_score(
    triggered_rules: list[dict[str, Any]],
    all_rules: list[dict[str, Any]],
    base_score: float = 100.0,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Calculate final score for a sample."""
    total_penalty = 0.0
    total_reward = 0.0
    dimension_breakdown = {}

    for triggered in triggered_rules:
        rule = triggered["rule"]
        score = rule["score"]
        weight = rule.get("weight", 1.0)
        impact = score * weight

        if score < 0:
            total_penalty += impact
        else:
            total_reward += impact

        dim = rule["rule_dimension"]
        if dim not in dimension_breakdown:
            dimension_breakdown[dim] = 0.0
        dimension_breakdown[dim] += impact

    final_score = max(min_score, base_score + total_penalty + total_reward)

    return {
        "base_score": base_score,
        "total_penalty": round(total_penalty, 2),
        "total_reward": round(total_reward, 2),
        "final_score": round(final_score, 2),
        "dimension_breakdown": {k: round(v, 2) for k, v in dimension_breakdown.items()},
        "triggered_rules_count": len(triggered_rules),
        "penalty_count": sum(1 for t in triggered_rules if t["rule"]["score"] < 0),
        "reward_count": sum(1 for t in triggered_rules if t["rule"]["score"] > 0),
    }


def _build_eval_prompt(
    dialog_context: str,
    response: str,
    rule: dict[str, Any],
    current_turn: int = 1,
    slot_status: Optional[dict[str, int]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Build evaluation prompt for a single rule."""
    config_params = rule.get("config", {})
    config_str = (
        json.dumps(config_params, ensure_ascii=False) if config_params else "无"
    )

    prompt = f"""你是一名专业对话质量评估员。请根据以下规则评估当前回复：

【规则信息】
- 规则：{rule["rule_name_cn"]} ({rule["rule_name"]})
- 维度：{rule["rule_dimension"]}
- 描述：{rule["description"]}
- 触发类型：{rule["trigger"]}
- 触发条件：{rule["trigger_condition"]}
- 配置参数：{config_str}

【对话上下文】
{dialog_context if dialog_context else "单轮对话，无历史上下文"}

【当前回复（待评估）】
{response}

⚠️ 重要说明：
1. 请只评估【当前回复】是否符合规则
2. 【对话上下文】仅供参考，帮助理解当前回复的背景
3. 请不要对【对话上下文】中的历史回复进行违规判断

【辅助信息】
- 当前轮次：{current_turn}
- 槽位状态：{json.dumps(slot_status, ensure_ascii=False) if slot_status else "未知"}
- System Prompt 要求：{system_prompt[:500] if system_prompt else "无"}...

 【输出要求】
请严格按以下 JSON Schema 输出评估结果（仅输出 JSON，无其他内容）：
{json.dumps(rule["kwargs_schema"], ensure_ascii=False, indent=2)}

注意：
1. ⚠️ 所有枚举值字段必须使用小写（例如：使用 "true" 而非 "True"，使用 "yes" 而非 "Yes"）
2. reason 字段简要说明判断依据（50 字以内）
3. 确保输出为合法 JSON，可直接解析
4. 如果字段不适用，result 填 null"""

    return prompt


def extract_final_response(text: str) -> str:
    """Extract content between BEGIN_FINAL and END_FINAL from GPT response."""
    if not text:
        return text

    if "BEGIN_FINAL" in text and "END_FINAL" in text:
        match = re.search(r"BEGIN_FINAL\s*(.*?)\s*END_FINAL", text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = extracted.replace("<sep>", " ")
            extracted = " ".join(extracted.split())
            return extracted

    return text


def extract_json_from_text(text: str) -> str:
    """从文本中提取 JSON 内容，支持 markdown 代码块包裹的 JSON"""
    if not text:
        return text

    # 尝试直接解析
    text = text.strip()
    try:
        json.loads(text)
        return text
    except:
        pass

    # 尝试提取 markdown 代码块中的 JSON
    # 匹配 ```json ... ``` 或 ``` ... ```
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            try:
                json.loads(json_content)
                return json_content
            except:
                pass

    # 如果都没找到，返回原始文本
    return text


def _build_single_call_eval_prompt(
    *,
    dialog_context: str,
    response: str,
    active_rules: list[dict[str, Any]],
    current_turn: int,
    slot_status: Optional[dict[str, int]],
    system_prompt: Optional[str],
) -> str:
    rule_blocks: list[str] = []
    for rule in active_rules:
        config_str = (
            json.dumps(rule.get("config", {}), ensure_ascii=False)
            if rule.get("config")
            else "无"
        )
        rule_blocks.append(
            f"- rule_id: {rule['rule_id']}\n"
            f"  rule_name: {rule['rule_name_cn']} ({rule['rule_name']})\n"
            f"  rule_type: {rule['rule_type']}\n"
            f"  rule_dimension: {rule['rule_dimension']}\n"
            f"  score: {rule['score']}\n"
            f"  trigger: {rule['trigger']}\n"
            f"  trigger_condition: {rule['trigger_condition']}\n"
            f"  config: {config_str}\n"
            f"  description: {rule['description']}"
        )

    active_rule_ids = [rule["rule_id"] for rule in active_rules]
    reward_rule_ids = [rule["rule_id"] for rule in active_rules if rule.get("rule_type") == "reward"]
    penalty_rule_ids = [rule["rule_id"] for rule in active_rules if rule.get("rule_type") != "reward"]

    return f"""你是一名专业对话质量评估员。请一次性完整检查所有 active rules，但只输出命中的规则。

【Active Rules】
{chr(10).join(rule_blocks)}

【对话上下文】
{dialog_context if dialog_context else "单轮对话，无历史上下文"}

【当前回复（待评估）】
{response}

【辅助信息】
- 当前轮次：{current_turn}
- 槽位状态：{json.dumps(slot_status, ensure_ascii=False) if slot_status else "未知"}
- System Prompt 要求：{system_prompt[:500] if system_prompt else "无"}...

【任务要求】
1. 只评估【当前回复】是否命中当前 active rules。
2. 【对话上下文】仅用于理解语境，不能把历史轮次内容直接判为当前轮命中。
3. 你必须完整检查全部 active_rule_ids：{active_rule_ids}
4. 对 penalty 规则，只有确认当前回复违规时，才写入 violations。
5. 对 reward 规则，只有确认当前回复达成奖励时，才写入 rewarded_rule_ids。
6. 没写入 violations 或 rewarded_rule_ids 的规则，表示你已检查且确认当前轮未命中。
7. covered_rule_ids 必须完整覆盖：{active_rule_ids}
8. violations 中每条都必须提供：
   - rule_id：且必须属于 penalty 规则 {penalty_rule_ids}
   - evidence：最多 30 个字，引用关键短语或概括关键行为
   - analysis：只允许 1 句话，20-40 个字，禁止提及其他规则、禁止输出思考过程
9. rewarded_rule_ids 中只能填写 reward 规则：{reward_rule_ids}
10. 不要输出 Markdown，不要输出额外解释，只输出 JSON。

【输出格式】
{{
  "covered_rule_ids": {active_rule_ids},
  "violations": [
    {{
      "rule_id": {penalty_rule_ids[0] if penalty_rule_ids else 1},
      "evidence": "当前回复中的关键短语",
      "analysis": "当前回复违反该规则的简短原因。"
    }}
  ],
  "rewarded_rule_ids": {reward_rule_ids if reward_rule_ids else []}
}}
"""


def _build_single_call_rule_evaluations(
    result_data: dict[str, Any], active_rules: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_ids = {int(rule["rule_id"]) for rule in active_rules}
    covered_ids = set(_normalize_rule_id_list(result_data.get("covered_rule_ids")))
    if covered_ids != expected_ids:
        missing = sorted(expected_ids - covered_ids)
        extra = sorted(covered_ids - expected_ids)
        raise ValueError(
            f"covered_rule_ids mismatch: missing={missing}, extra={extra}"
        )

    reward_rule_ids = {
        int(rule["rule_id"]) for rule in active_rules if rule.get("rule_type") == "reward"
    }
    penalty_rule_ids = expected_ids - reward_rule_ids

    rewarded_rule_ids = set(_normalize_rule_id_list(result_data.get("rewarded_rule_ids")))
    invalid_reward_ids = sorted(rewarded_rule_ids - reward_rule_ids)
    if invalid_reward_ids:
        raise ValueError(f"rewarded_rule_ids contain non-reward rules: {invalid_reward_ids}")

    raw_violations = result_data.get("violations")
    if not isinstance(raw_violations, list):
        raw_violations = []

    violations_by_id: dict[int, dict[str, str]] = {}
    for item in raw_violations:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("rule_id"))
        except (TypeError, ValueError):
            continue
        if rid not in penalty_rule_ids:
            raise ValueError(f"violations contain non-penalty rule_id={rid}")
        evidence = _sanitize_violation_evidence(item.get("evidence", ""))
        analysis = _sanitize_violation_analysis(item.get("analysis", ""))
        violations_by_id[rid] = {
            "evidence": evidence,
            "analysis": analysis or "模型确认该规则命中。",
        }

    rule_evaluations: list[dict[str, Any]] = []
    for rule in active_rules:
        rid = int(rule["rule_id"])
        if rid in violations_by_id:
            eval_result = {
                "is_violation": "true",
                "evidence": violations_by_id[rid]["evidence"],
                "analysis": violations_by_id[rid]["analysis"],
                "single_call_protocol": "violations_only",
            }
            triggered = True
        elif rid in rewarded_rule_ids:
            eval_result = {
                "is_rewarded": "true",
                "single_call_protocol": "violations_only",
            }
            triggered = True
        else:
            eval_result = {
                "single_call_protocol": "violations_only",
            }
            triggered = False

        rule_evaluations.append(
            {
                "rule_id": rid,
                "rule_name": rule["rule_name"],
                "rule_name_cn": rule["rule_name_cn"],
                "rule_dimension": rule["rule_dimension"],
                "triggered": triggered,
                "eval_result": eval_result,
                "config_used": rule.get("config", {}),
                "error": None,
            }
        )

    return rule_evaluations


async def _evaluate_single_rule(
    rule: dict[str, Any],
    dialog_context: str,
    response: str,
    current_turn: int,
    slot_status: Optional[dict[str, int]],
    system_prompt: Optional[str],
    reward_router_address: str,
    judge_model: str,
    timeout_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    verbose: bool = False,
    trace_enabled: bool = False,
    trace_path: str = "",
    trace_common: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate a single rule via LLM API."""
    prompt = _build_eval_prompt(
        dialog_context, response, rule, current_turn, slot_status, system_prompt
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    max_retries = max(1, int(max_retries))
    retry_delay = max(0.0, float(retry_backoff_s))
    result_text = ""  # 初始化变量

    def _trace_record(error: Optional[str]) -> None:
        if not trace_enabled:
            return
        _append_genrm_trace(
            trace_path,
            {
                "ts": time.time(),
                "event": "genrm_io",
                "rm_name": "instruction_reward",
                "stage": "single_rule_eval",
                "rule_id": rule.get("rule_id"),
                "rule_name": rule.get("rule_name"),
                "judge_model": judge_model,
                "request_prompt": prompt,
                "response_text": result_text,
                "error": error,
                "meta": trace_common or {},
            },
        )

    for attempt in range(max_retries):
        try:
            resp = await _chat_complete(
                reward_router_address=reward_router_address,
                payload=payload,
                timeout_s=timeout_s,
                api_key=api_key,
            )

            result_text = (
                resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # 从可能包含 markdown 的文本中提取 JSON
            result_text = extract_json_from_text(result_text)

            if not result_text or not result_text.strip():
                if attempt < max_retries - 1:
                    if verbose:
                        print(
                            f"      ⚠️ 规则 {rule['rule_id']}: LLM 返回空响应，重试 ({attempt + 1}/{max_retries})"
                        )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    if verbose:
                        print(
                            f"      ✗ 规则 {rule['rule_id']}: LLM 返回空响应，已达最大重试次数"
                        )
                    _trace_record("Empty response from LLM")
                    return {
                        "rule_id": rule.get("rule_id", "unknown"),
                        "rule_name": rule.get("rule_name", "unknown"),
                        "rule_name_cn": rule.get("rule_name_cn", "unknown"),
                        "rule_dimension": rule.get("rule_dimension", ""),
                        "triggered": False,
                        "eval_result": {},
                        "config_used": rule.get("config", {}),
                        "error": "Empty response from LLM",
                    }

            result = json.loads(result_text)
            result = _normalize_json_bools(result)

            is_triggered = check_rule_triggered(
                eval_result=result,
                rule=rule,
                current_turn=current_turn,
                slot_status=slot_status,
            )
            _trace_record(None)

            return {
                "rule_id": rule["rule_id"],
                "rule_name": rule["rule_name"],
                "rule_name_cn": rule["rule_name_cn"],
                "rule_dimension": rule["rule_dimension"],
                "triggered": is_triggered,
                "eval_result": result,
                "config_used": rule.get("config", {}),
                "error": None,
            }

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(
                        f"      ⚠️ 规则 {rule['rule_id']}: JSON 解析失败，重试 ({attempt + 1}/{max_retries}) - 响应: {result_text[:100] if result_text else '(empty)'}"
                    )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                if verbose:
                    print(
                        f"      ✗ 规则 {rule['rule_id']}: JSON 解析失败 - 响应: {result_text[:100] if result_text else '(empty)'}"
                    )
                _trace_record(f"JSON decode error: {str(e)}")
                return {
                    "rule_id": rule.get("rule_id", "unknown"),
                    "rule_name": rule.get("rule_name", "unknown"),
                    "rule_name_cn": rule.get("rule_name_cn", "unknown"),
                    "rule_dimension": rule.get("rule_dimension", ""),
                    "triggered": False,
                    "eval_result": {},
                    "config_used": rule.get("config", {}),
                    "error": f"JSON decode error: {str(e)}",
                }

        except Exception as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(
                        f"      ⚠️ 规则 {rule['rule_id']}: 请求异常，重试 ({attempt + 1}/{max_retries}) - {type(e).__name__}: {str(e)}"
                    )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                if verbose:
                    print(
                        f"      ✗ 规则 {rule['rule_id']}: 请求失败 - {type(e).__name__}: {str(e)}"
                    )
                _trace_record(str(e))
                return {
                    "rule_id": rule.get("rule_id", "unknown"),
                    "rule_name": rule.get("rule_name", "unknown"),
                    "rule_name_cn": rule.get("rule_name_cn", "unknown"),
                    "rule_dimension": rule.get("rule_dimension", ""),
                    "triggered": False,
                    "eval_result": {},
                    "config_used": rule.get("config", {}),
                    "error": str(e),
                }

    _trace_record("Max retries exceeded")
    return {
        "rule_id": rule.get("rule_id", "unknown"),
        "rule_name": rule.get("rule_name", "unknown"),
        "error": "Max retries exceeded",
        "triggered": False,
    }


async def _evaluate_rules_single_call(
    *,
    active_rules: list[dict[str, Any]],
    dialog_context: str,
    response: str,
    current_turn: int,
    slot_status: Optional[dict[str, int]],
    system_prompt: Optional[str],
    reward_router_address: str,
    judge_model: str,
    timeout_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    verbose: bool = False,
    trace_enabled: bool = False,
    trace_path: str = "",
    trace_common: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    prompt = _build_single_call_eval_prompt(
        dialog_context=dialog_context,
        response=response,
        active_rules=active_rules,
        current_turn=current_turn,
        slot_status=slot_status,
        system_prompt=system_prompt,
    )
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    max_retries = max(1, int(max_retries))
    retry_delay = max(0.0, float(retry_backoff_s))
    result_text = ""

    def _trace_record(error: Optional[str]) -> None:
        if not trace_enabled:
            return
        _append_genrm_trace(
            trace_path,
            {
                "ts": time.time(),
                "event": "genrm_io",
                "rm_name": "instruction_reward",
                "stage": "active_rules_single_call_eval",
                "judge_model": judge_model,
                "active_rule_ids": [rule.get("rule_id") for rule in active_rules],
                "request_prompt": prompt,
                "response_text": result_text,
                "error": error,
                "meta": trace_common or {},
            },
        )

    for attempt in range(max_retries):
        try:
            resp = await _chat_complete(
                reward_router_address=reward_router_address,
                payload=payload,
                timeout_s=timeout_s,
                api_key=api_key,
            )
            result_text = (
                resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            result_text = extract_json_from_text(result_text)
            if not result_text or not result_text.strip():
                raise ValueError("Empty response from LLM")
            result_data = json.loads(result_text)
            rule_evaluations = _build_single_call_rule_evaluations(result_data, active_rules)
            _trace_record(None)
            return rule_evaluations
        except Exception as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(
                        f"  ⚠️ 单次 active-rules 评估失败，重试 ({attempt + 1}/{max_retries}) - {type(e).__name__}: {str(e)}"
                    )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                _trace_record(str(e))
                raise

    raise RuntimeError("single_call_max_retries_exceeded")


async def _evaluate_rules_individually(
    *,
    active_rules: list[dict[str, Any]],
    dialog_context: str,
    response: str,
    current_turn: int,
    slot_status: Optional[dict[str, int]],
    system_prompt: Optional[str],
    reward_router_address: str,
    judge_model: str,
    timeout_s: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    verbose: bool = False,
    trace_enabled: bool = False,
    trace_path: str = "",
    trace_common: Optional[dict[str, Any]] = None,
    eval_concurrency: int = 5,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(eval_concurrency)

    async def _evaluate_with_semaphore(rule: dict[str, Any], idx: int) -> dict[str, Any]:
        async with semaphore:
            if verbose:
                print(
                    f"    [{idx + 1}/{len(active_rules)}] 开始评估规则 {rule['rule_id']}: {rule.get('rule_name_cn', rule.get('rule_name', 'unknown'))}"
                )
            try:
                result = await _evaluate_single_rule(
                    rule=rule,
                    dialog_context=dialog_context,
                    response=response,
                    current_turn=current_turn,
                    slot_status=slot_status,
                    system_prompt=system_prompt,
                    reward_router_address=reward_router_address,
                    judge_model=judge_model,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    verbose=verbose,
                    trace_enabled=trace_enabled,
                    trace_path=trace_path,
                    trace_common=trace_common,
                )
                if verbose:
                    print(
                        f"    [{idx + 1}/{len(active_rules)}] ✓ 规则 {rule['rule_id']} 完成 - 触发: {result.get('triggered', False)} - 错误: {result.get('error', 'None')}"
                    )
                return result
            except Exception as e:
                if verbose:
                    print(
                        f"    [{idx + 1}/{len(active_rules)}] ✗ 规则 {rule['rule_id']} 异常: {type(e).__name__}: {str(e)}"
                    )
                return {
                    "rule_id": rule.get("rule_id", "unknown"),
                    "rule_name": rule.get("rule_name", "unknown"),
                    "rule_name_cn": rule.get("rule_name_cn", "unknown"),
                    "rule_dimension": rule.get("rule_dimension", ""),
                    "triggered": False,
                    "eval_result": {},
                    "config_used": rule.get("config", {}),
                    "error": f"{type(e).__name__}: {str(e)}",
                }

    if verbose:
        print(f"  开始并发评估 {len(active_rules)} 条规则（并发限制: {eval_concurrency}）...")
    tasks = [_evaluate_with_semaphore(rule, idx) for idx, rule in enumerate(active_rules)]
    return await asyncio.gather(*tasks)


async def compute_instruction_score(
    question: str,
    answer: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None = None,
    rubrics: list[dict[str, Any]] | None = None,
    rubrics_path: str | None = None,
    enable_rule_mapper: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute instruction-following score using multi-rule evaluation.

    This function loads rubrics rules, optionally filters them using LLM-based
    rule mapping based on system_prompt, then evaluates each active rule and
    aggregates the final score.

    Args:
        question: User question/input (used as dialog context start)
        answer: Model response to evaluate
        ground_truth: Ground truth answer (not used in current implementation)
        extra_info: Additional info containing history, system_prompt, slot_status, etc.
        reward_router_address: Address for reward router API (host:port)
        rubrics: List of rule configurations for evaluation (optional)
        rubrics_path: Path to rubrics JSON/JSONL file (optional)
        enable_rule_mapper: Whether to enable LLM-based rule filtering (default: True)
        **kwargs: Additional parameters:
            - judge_model: Model name for evaluation (default: "qwen-plus")
            - judge_timeout_s: Timeout for each API call (default: 60.0)
            - judge_temperature: Temperature for generation (default: 0.1)
            - judge_max_tokens: Max tokens for response (default: 512)
            - current_turn: Current turn number (default: 1)
            - base_score: Base score for calculation (default: 100.0)
            - instruction_eval_mode: "single_call" or "per_rule" (default: "single_call")

    Returns:
        dict with keys:
            - score: Normalized score in [0, 1]
            - raw: Raw response summary (JSON string)
            - status: Status string ("ok" or error type)
            - triggered_rules: List of triggered rule details
            - dimension_breakdown: Score breakdown by dimension
            - rule_evaluations: All rule evaluation results
            - active_rules: List of active rule IDs (after mapping)
            - rule_mapping_info: Rule mapping metadata (if enabled)

    Example:
        >>> result = await compute_instruction_score(
        ...     question="你好",
        ...     answer="你好，有什么可以帮您的吗？",
        ...     ground_truth="",
        ...     extra_info={"history": [], "system_prompt": "..."},
        ...     reward_router_address="localhost:8000",
        ...     rubrics_path="./rubrics.json",
        ...     enable_rule_mapper=True,
        ...     judge_model="qwen-plus",
        ... )
        >>> print(result["score"])  # 0.0 - 1.0
    """
    del ground_truth

    if not reward_router_address:
        return {"score": 0.0, "raw": "", "status": "reward_router_address_missing"}

    if rubrics is None and rubrics_path is None:
        rubrics_path = str(
            kwargs.get("instruction_rubrics_path")
            or kwargs.get("rubrics_path")
            or DEFAULT_RUBRICS_PATH
        )

    final_rubrics: list[dict[str, Any]]
    if rubrics is not None:
        final_rubrics = rubrics
    elif rubrics_path is not None:
        try:
            final_rubrics = load_rubrics(rubrics_path)
        except Exception as e:
            return {
                "score": 0.0,
                "raw": "",
                "status": f"rubrics_load_error:{str(e)}",
            }
    else:
        return {"score": 0.0, "raw": "", "status": "rubrics_missing"}

    judge_model = (
        kwargs.get("instruction_judge_model")
        or kwargs.get("judge_model")
        or "qwen-plus"
    )
    timeout_s = _to_float(
        kwargs.get("instruction_judge_timeout_s", kwargs.get("judge_timeout_s", 120.0)),
        default=120.0,
    )
    temperature = _to_float(
        kwargs.get("instruction_judge_temperature", kwargs.get("judge_temperature", 0.1)),
        default=0.1,
    )
    top_p = _to_float(
        kwargs.get("instruction_judge_top_p", kwargs.get("judge_top_p", 1.0)),
        default=1.0,
    )
    max_tokens = _to_int(
        kwargs.get("instruction_judge_max_tokens", kwargs.get("judge_max_tokens", 4096)),
        default=4096,
    )
    max_retries = max(
        1,
        _to_int(
            kwargs.get(
                "instruction_judge_max_retries",
                kwargs.get("judge_max_retries", 3),
            ),
            default=3,
        ),
    )
    retry_backoff_s = max(
        0.0,
        _to_float(
            kwargs.get(
                "instruction_judge_retry_backoff_s",
                kwargs.get("judge_retry_backoff_s", 1.0),
            ),
            default=1.0,
        ),
    )
    base_score = _to_float(
        kwargs.get("instruction_base_score", kwargs.get("base_score", 100.0)),
        default=100.0,
    )
    api_key = (
        kwargs.get("instruction_api_key")
        or kwargs.get("api_key")
        or kwargs.get("llm_api_key")
    )
    verbose = _to_bool(
        kwargs.get("instruction_reward_verbose", kwargs.get("reward_debug", False)),
        default=False,
    )
    eval_concurrency = max(
        1,
        _to_int(kwargs.get("instruction_rule_eval_concurrency", 5), default=5),
    )
    trace_enabled = _to_bool(kwargs.get("collect_genrm_io", False), default=False)
    trace_path = str(kwargs.get("genrm_io_path", "/tmp/genrm_io.jsonl"))
    eval_mode = str(kwargs.get("instruction_eval_mode", "single_call")).strip().lower()
    single_call_fallback = _to_bool(
        kwargs.get("instruction_single_call_fallback_to_per_rule", True),
        default=True,
    )
    enable_rule_mapper = _to_bool(
        kwargs.get("instruction_enable_rule_mapper", enable_rule_mapper),
        default=enable_rule_mapper,
    )

    dialog_context = ""
    system_prompt = ""
    slot_status = {}
    current_turn: int | None = None
    trace_meta: dict[str, Any] = {}

    if isinstance(extra_info, dict):
        history_list = _extract_history_list(extra_info)
        if isinstance(history_list, list):
            context_lines = []
            for msg in history_list:
                normalized = _normalize_history_turn(msg)
                if normalized is None:
                    continue
                from_role, clean_content = normalized
                context_lines.append(f"{from_role}: {clean_content}")
            dialog_context = "\n".join(context_lines)

        if question:
            clean_question = _clean_context_text(str(question))
            question_line = f"用户: {clean_question}"
            if clean_question and (
                not dialog_context or not dialog_context.splitlines() or dialog_context.splitlines()[-1] != question_line
            ):
                dialog_context = f"{dialog_context}\n{question_line}".strip()

        system_prompt = _extract_system_prompt(extra_info)
        raw_slot_status = extra_info.get("slot_status", {})
        slot_status = raw_slot_status if isinstance(raw_slot_status, dict) else {}
        current_turn = _to_int(
            extra_info.get("turn_round", extra_info.get("current_turn", None)),
            default=1,
        )
        trace_meta = {
            "sample_id": extra_info.get("sample_id"),
            "source": extra_info.get("source"),
            "conv_id": extra_info.get("conv_id"),
            "turn_id": extra_info.get("turn_id"),
        }

    if "current_turn" in kwargs:
        current_turn = _to_int(kwargs.get("current_turn"), default=current_turn or 1)
    if current_turn is None or current_turn <= 0:
        current_turn = 1

    rule_mapping_info = {
        "is_dynamic": False,
        "total_rules": len(final_rubrics),
        "active_rules": len(final_rubrics),
        "mapping_error": None,
        "eval_mode": eval_mode,
    }

    active_rules = final_rubrics

    if enable_rule_mapper and system_prompt and system_prompt.strip():
        try:
            mapper = RuleMapper(
                reward_router_address=reward_router_address,
                judge_model=judge_model,
                timeout_s=timeout_s,
                temperature=temperature,
                api_key=api_key,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                verbose=verbose,
            )
            active_rules = await mapper.map_rules(system_prompt, final_rubrics)
            rule_mapping_info["is_dynamic"] = True
            rule_mapping_info["active_rules"] = len(active_rules)
            if not active_rules:
                rule_mapping_info["mapping_error"] = "rule_mapper_returned_empty"
                active_rules = final_rubrics
                rule_mapping_info["active_rules"] = len(active_rules)
        except Exception as e:
            rule_mapping_info["mapping_error"] = str(e)
            active_rules = final_rubrics
    else:
        if verbose:
            print("  [DEBUG] 规则映射未启用或 system_prompt 为空")
            print(f"           - enable_rule_mapper: {enable_rule_mapper}")
            print(
                f"           - system_prompt 存在: {bool(system_prompt and system_prompt.strip())}"
            )

    try:
        if eval_mode == "single_call":
            if verbose:
                print(f"  开始单次评估 {len(active_rules)} 条 active rules...")
            try:
                rule_evaluations = await _evaluate_rules_single_call(
                    active_rules=active_rules,
                    dialog_context=dialog_context,
                    response=answer,
                    current_turn=current_turn,
                    slot_status=slot_status if slot_status else None,
                    system_prompt=system_prompt if system_prompt else None,
                    reward_router_address=reward_router_address,
                    judge_model=judge_model,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    verbose=verbose,
                    trace_enabled=trace_enabled,
                    trace_path=trace_path,
                    trace_common=trace_meta,
                )
                if verbose:
                    print("  ✓ 单次 active-rules 评估完成")
            except Exception as exc:
                if not single_call_fallback:
                    raise
                if verbose:
                    print(
                        f"  ⚠️ 单次 active-rules 评估失败，回退到逐规则评估 - {type(exc).__name__}: {str(exc)}"
                    )
                rule_mapping_info["eval_mode"] = "per_rule_fallback"
                rule_evaluations = await _evaluate_rules_individually(
                    active_rules=active_rules,
                    dialog_context=dialog_context,
                    response=answer,
                    current_turn=current_turn,
                    slot_status=slot_status if slot_status else None,
                    system_prompt=system_prompt if system_prompt else None,
                    reward_router_address=reward_router_address,
                    judge_model=judge_model,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    verbose=verbose,
                    trace_enabled=trace_enabled,
                    trace_path=trace_path,
                    trace_common=trace_meta,
                    eval_concurrency=eval_concurrency,
                )
        else:
            if verbose:
                print(f"  开始并发评估 {len(active_rules)} 条规则（并发限制: {eval_concurrency}）...")
            rule_evaluations = await _evaluate_rules_individually(
                active_rules=active_rules,
                dialog_context=dialog_context,
                response=answer,
                current_turn=current_turn,
                slot_status=slot_status if slot_status else None,
                system_prompt=system_prompt if system_prompt else None,
                reward_router_address=reward_router_address,
                judge_model=judge_model,
                timeout_s=timeout_s,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                api_key=api_key,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                verbose=verbose,
                trace_enabled=trace_enabled,
                trace_path=trace_path,
                trace_common=trace_meta,
                eval_concurrency=eval_concurrency,
            )
            if verbose:
                print("  ✓ 所有规则评估完成")

    except Exception as exc:
        return {
            "score": 0.0,
            "raw": "",
            "status": f"evaluation_error:{type(exc).__name__}",
            "triggered_rules": [],
            "dimension_breakdown": {},
            "rule_evaluations": [],
            "active_rules": [r["rule_id"] for r in active_rules],
            "rule_mapping_info": rule_mapping_info,
        }

    triggered_rules = []
    valid_rule_evaluations = []

    for rule_eval in rule_evaluations:
        # 过滤掉异常对象
        if isinstance(rule_eval, Exception):
            continue
        valid_rule_evaluations.append(rule_eval)

        if rule_eval.get("triggered") and rule_eval.get("error") is None:
            rule = next(
                (r for r in active_rules if r["rule_id"] == rule_eval["rule_id"]), None
            )
            if rule:
                triggered_rules.append(
                    {
                        "rule": rule,
                        "eval_result": rule_eval.get("eval_result", {}),
                    }
                )

    # 使用过滤后的评估结果
    rule_evaluations = valid_rule_evaluations
    eval_total = len(rule_evaluations)
    eval_error_count = sum(1 for item in rule_evaluations if item.get("error"))

    if eval_total > 0 and eval_error_count == eval_total:
        raw_summary = json.dumps(
            {
                "rule_evaluations": [
                    {
                        "rule_id": r["rule_id"],
                        "triggered": r.get("triggered", False),
                        "error": r.get("error"),
                    }
                    for r in rule_evaluations
                ]
            },
            ensure_ascii=False,
        )
        result_payload = {
            "score": 0.0,
            "raw": raw_summary,
            "status": "evaluation_all_failed",
            "triggered_rules": [],
            "dimension_breakdown": {},
            "rule_evaluations": rule_evaluations,
            "active_rules": [
                {
                    "rule_id": r["rule_id"],
                    "rule_name": r["rule_name"],
                    "rule_name_cn": r["rule_name_cn"],
                }
                for r in active_rules
            ],
            "rule_mapping_info": rule_mapping_info,
            "error_count": eval_error_count,
        }
        if trace_enabled:
            _append_genrm_trace(
                trace_path,
                {
                    "ts": time.time(),
                    "event": "genrm_summary",
                    "rm_name": "instruction_reward",
                    "status": result_payload["status"],
                    "score": result_payload["score"],
                    "error_count": eval_error_count,
                    "question": question,
                    "answer": answer,
                    "meta": trace_meta,
                },
            )
        return result_payload

    score_result = calculate_sample_score(triggered_rules, active_rules, base_score)

    normalized_score = round(score_result["final_score"] / 100.0, 2)

    triggered_rules_detail = [
        {
            "rule_id": t["rule"]["rule_id"],
            "rule_name": t["rule"]["rule_name"],
            "rule_name_cn": t["rule"]["rule_name_cn"],
            "rule_dimension": t["rule"]["rule_dimension"],
            "score": t["rule"]["score"],
            "weight": t["rule"].get("weight", 1.0),
            "penalty": t["rule"]["score"] * t["rule"].get("weight", 1.0),
            "eval_detail": t["eval_result"],
            "config_used": t["rule"].get("config", {}),
        }
        for t in triggered_rules
    ]

    raw_summary = json.dumps(
        {
            "rule_evaluations": [
                {
                    "rule_id": r["rule_id"],
                    "triggered": r.get("triggered", False),
                    "error": r.get("error"),
                }
                for r in rule_evaluations
            ]
        },
        ensure_ascii=False,
    )

    result_payload = {
        "score": normalized_score,
        "raw": raw_summary,
        "status": "ok_with_errors" if eval_error_count > 0 else "ok",
        "triggered_rules": triggered_rules_detail,
        "dimension_breakdown": score_result["dimension_breakdown"],
        "rule_evaluations": rule_evaluations,
        "active_rules": [
            {
                "rule_id": r["rule_id"],
                "rule_name": r["rule_name"],
                "rule_name_cn": r["rule_name_cn"],
            }
            for r in active_rules
        ],
        "rule_mapping_info": rule_mapping_info,
        "error_count": eval_error_count,
    }
    if trace_enabled:
        _append_genrm_trace(
            trace_path,
            {
                "ts": time.time(),
                "event": "genrm_summary",
                "rm_name": "instruction_reward",
                "status": result_payload["status"],
                "score": result_payload["score"],
                "error_count": eval_error_count,
                "question": question,
                "answer": answer,
                "meta": trace_meta,
            },
        )
    return result_payload
