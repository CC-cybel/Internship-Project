#!/usr/bin/env python3
"""Distill stronger contact-stage responses with Claude for age-directed RL JSONL.

Input rows are the RL-style records produced under this directory:
  - prompt: [{role, content}, ...]
  - ground_truth: final assistant response only

The script asks Claude to generate a more targeted, benefit-forward final reply
that follows the row's system prompt and conversation history. The original
final reply is intentionally not sent to the distillation model. The script
writes the same row shape back out, replacing ground_truth with the distilled
response and storing the original target in extra_info for traceability.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import httpx
except Exception:
    httpx = None
from openai import OpenAI
from tqdm import tqdm

DEFAULT_INPUT = "/data/chengch/project/data_remake/claude_distill/single_turn_rl_contact_age_directed.all.jsonl"
DEFAULT_OUTPUT = "/data/chengch/project/data_remake/claude_distill/single_turn_rl_contact_age_directed.claude_distilled.llama_factory.json"
DEFAULT_RAW_LOG = "/data/chengch/project/data_remake/claude_distill/logs/claude_distill_contact_age_directed_v4_no_target_raw.txt"
DEFAULT_CACHE_DIR = "/data/chengch/project/data_remake/claude_distill/cache/contact_age_directed_v4_no_target"

API_KEY = os.environ.get("CLAUDE_DISTILL_API_KEY", "")
BASE_URL = os.environ.get("CLAUDE_DISTILL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.environ.get("CLAUDE_DISTILL_MODEL_NAME", "deepseek-v4-flash")
REQUEST_TIMEOUT = float(os.environ.get("CLAUDE_DISTILL_TIMEOUT", "300"))

MAX_RETRIES = 3
DEFAULT_REPAIR_ROUNDS = 2
RETRY_BASE_SECONDS = 1.0
RETRY_JITTER_SECONDS = 0.5

FORBIDDEN_OUTPUT_MARKERS = ("BEGIN_META", "END_META", "BEGIN_FINAL", "END_FINAL")
CONTACT_WORD_RE = re.compile(r"电话|手机号|联系方式|微信|回电|加.{0,4}微信|联系")
REQUIRED_SLOT_KEYS = [
    "age",
    "gender",
    "name",
    "phone",
    "wechat",
    "symptom",
    "duration",
    "medical_history",
    "relationship",
    "medical_awareness",
]

SYSTEM_PROMPT = """
你是资深医疗咨询专家。你会看见一段对话，其中包含 system prompt 和截至最后一个用户输入为止的历史对话。你的任务是直接生成新的最后一轮 assistant 回复，作为蒸馏目标。你不会看到原始最后一轮 assistant 回复，必须只根据 system prompt 和历史对话独立生成。

你必须只输出一段完整的回复文本，不要输出 JSON，不要输出 Markdown，不要解释。输出格式必须严格如下：
BEGIN_META
action=...
thought=...
slot_age=...
slot_gender=...
slot_name=...
slot_phone=...
slot_wechat=...
slot_symptom=...
slot_duration=...
slot_medical_history=...
slot_relationship=...
slot_medical_awareness=...
END_META
BEGIN_FINAL
用户可见回复
END_FINAL

任务目标：
1. 严格遵守输入样本里的 system prompt，包括留联触发轮次、分龄联系方式策略、询问密度、语言风格、轻量用户模型、槽位表和拒绝处理。
2. 当前样本处于留联/套联阶段，BEGIN_FINAL 要比原始回复更针对用户当前核心诉求，福利感更强，让用户觉得留下电话或微信能获得现实帮助。
3. 福利感自然可信，可以围绕免费初步评估、专科老师一对一指导、方案梳理、用药或风险核对、就医路径、复诊复学建议、家属照护建议等展开。
4. 不得编造未在 system prompt 或历史中出现的具体医院资质、价格承诺、疗效保证；不要承诺一定安排、一定有效、一定治好。
5. 若用户表现出自伤、自杀、极端绝望等危机风险，必须先做安全承接，再用危机安全回电或专业评估为理由留联。
6. 若已知用户年龄大于 35 岁，优先索要电话；若年龄小于等于 35 岁或未知，优先索要微信。若用户已提供某种联系方式，则确认并说明后续帮助。
7. 回复要像真人医疗咨询，不要客服腔，不要列表。
8. 最后用户可见的回复字数在60-90字左右。

BEGIN_META 要求：
- action 只能继承最后一个 user 输入中的 <action>，历史轮次里的 <action> 不能继承；最后一个 user 没有 <action> 时必须写 action=。
- thought 必须写在 thought= 后面的同一行，不要在 thought 内换行，不要把分析字段拆到下一行。
- thought 必须严格按这个顺序输出四段：
  【锚定】当前轮次、用户关键信息、触发规则或动作依据。
  【分析】留联分层：user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。用户状态：...。生理层：...。心理层：...。战术层：...。
  【检验】检查是否符合留联轮次、分龄策略、询问密度、槽位和安全红线。
  【策略】说明本轮最终话术策略和索要电话/微信的理由。
- 【分析】内部字段顺序必须固定为：留联分层、用户状态、生理层、心理层、战术层；字段名必须使用中文冒号，不要写成等号。
- 留联分层必须按固定结构写：user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。
- fine_label 必须等于 user_type-core_need-conversion_barrier-lead_strategy。
- 生理层、心理层、战术层各只选择一个值并简短说明，不要并列多个层级或新增其他层级。
- 槽位必须使用具体值或兜底值，严禁 0/1。
- slot_* 中严禁出现 <sep>。thought 中也不可以使用 <sep>，用句号自然连接。
- 不要在 thought 中提到改写、蒸馏、Claude、模型、数据集、JSON 等加工痕迹。

槽位兜底：
- slot_age 未知写 未知
- slot_gender 写 男/女/未知
- slot_name 没有写 暂无
- slot_phone 未获取写 未获取
- slot_wechat 未获取写 未获取
- slot_symptom 未知写 未知
- slot_duration 没有写 暂无
- slot_medical_history 没有写 暂无
- slot_relationship 写 本人/母亲/父亲/伴侣/子女/朋友/其他家属/未知
- slot_medical_awareness 写 未知/小白/半懂/专业/误区明显

枚举：
user_type: [青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知]
core_need: [病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他]
conversion_barrier: [医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足]
lead_strategy: [低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊]
用户状态: [平静, 犹豫, 害怕, 不信任, 对抗, 急迫, 配合, 敷衍, 未知]
生理层: [皮质醇, 多巴胺, 内啡肽, 催产素, 血清素]，必须写 ↑ 或 ↓
心理层: [自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶]
战术层: [医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑]

BEGIN_FINAL 要求：
- 只写用户可见回复，可以使用 <sep> 分隔自然段，但不要滥用。
- 必须有明确但自然的留联动作，除非历史显示已经获取联系方式。
- 不要使用 Markdown，不要列表化，不要输出 BEGIN_META 之外的槽位。
回复完后必须有END_FINAL标记，且不能有多余文本。

一个可以参考的回复格式为：
BEGIN_META\naction=\nthought=【锚定】第14轮，无action。用户询问医生身份。【分析】留联分层：user_type=成人本人；core_need=治疗方案；conversion_barrier=信任顾虑；lead_strategy=科普判断转留联；fine_label=成人本人-治疗方案-信任顾虑-科普判断转留联。用户状态：不信任，用户对医生身份有疑虑。生理层：催产素↑，需要建立信任感。心理层：安全感，用户需要确认对方是正规医生。战术层：权威借势，强调医院正规性和医生资质。【检验】需回应信任顾虑。【策略】先回应信任顾虑，再引导用户等待电话。\nslot_age=33岁\nslot_gender=未知\nslot_name=陈\nslot_phone=18233081892\nslot_wechat=未获取\nslot_symptom=腿脚碰到东西就怀疑被动物碰到或咬到，每天恐惧害怕\nslot_duration=近两个月\nslot_medical_history=暂无\nslot_relationship=本人\nslot_medical_awareness=半懂\nEND_META\nBEGIN_FINAL\n我是石家庄长江心理精神医院的医生，我们医院是市医保单位、卫生局审批的正规医院，你可以放心。稍后会有医生联系你，先电话沟通下，再加你微信后期有问题你可以微信咨询。\nEND_FINAL
""".strip()

thread_local = threading.local()
log_lock = threading.Lock()
cache_lock = threading.Lock()


def get_client(args: argparse.Namespace) -> OpenAI:
    client = getattr(thread_local, "client", None)
    if client is None:
        client_kwargs = {"api_key": args.api_key, "base_url": args.base_url, "timeout": args.timeout}
        if httpx is not None:
            client_kwargs["http_client"] = httpx.Client(trust_env=False, timeout=args.timeout)
        client = OpenAI(**client_kwargs)
        thread_local.client = client
    return client


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Claude-distilled contact-stage responses for RL JSONL rows.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-format", choices=["llama_factory", "rl_jsonl"], default="llama_factory")
    parser.add_argument("--raw-log", default=DEFAULT_RAW_LOG)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条，0 表示处理全部。")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--repair-rounds", type=int, default=DEFAULT_REPAIR_ROUNDS, help="格式校验失败后的修复轮数，调试时可设为 0。")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="允许部分样本失败；输出成功样本，并将失败信息写入 failed jsonl。")
    parser.add_argument("--failed-output", default="", help="失败样本日志路径，默认使用 <output>.failed.jsonl。")
    return parser.parse_args()


def load_rows(path: str, offset: int, limit: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            if line_no <= offset:
                continue
            rows.append((line_no, json.loads(line)))
            if limit and len(rows) >= limit:
                break
    return rows


def clean_formatted_reply(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("BEGIN_META")
    end = text.rfind("END_FINAL")
    if start >= 0 and end >= start:
        text = text[start:end + len("END_FINAL")]

    # Normalize whitespace that models often introduce around structural markers.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{2,}(?=BEGIN_FINAL)", "\n", text)
    text = re.sub(r"(?<=END_META)\n{2,}", "\n", text)
    text = re.sub(r"(?<=BEGIN_META)\n{2,}", "\n", text)
    text = re.sub(r"\n{2,}(?=END_FINAL$)", "\n", text)
    return text.strip()


def parse_meta_slots(formatted: str) -> dict[str, str]:
    meta = formatted.split("END_META", 1)[0]
    slots: dict[str, str] = {}
    for line in meta.splitlines():
        if line.startswith("slot_") and "=" in line:
            key, value = line.split("=", 1)
            slots[key.removeprefix("slot_")] = value.strip()
    return slots


def expected_action_from_row(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") or []
    for message in reversed(prompt):
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content", ""))
            match = re.search(r"<action>(.*?)</action>", content)
            return match.group(1).strip() if match else ""
    return ""


def parse_action(formatted: str) -> str:
    for line in formatted.splitlines():
        if line.startswith("action="):
            return line.split("=", 1)[1].strip()
    return ""


def validate_formatted_reply(formatted: str, expected_action: str | None = None) -> str | None:
    text = clean_formatted_reply(formatted)
    pattern = re.compile(
        r"^BEGIN_META\naction=.*?\nthought=(?P<thought>[\s\S]*?)\n"
        r"slot_age=.*?\nslot_gender=.*?\nslot_name=.*?\nslot_phone=.*?\nslot_wechat=.*?\n"
        r"slot_symptom=.*?\nslot_duration=.*?\nslot_medical_history=.*?\n"
        r"slot_relationship=.*?\nslot_medical_awareness=.*?\n"
        r"END_META\nBEGIN_FINAL\n(?P<final>[\s\S]+?)\nEND_FINAL$"
    )
    match = pattern.match(text)
    if not match:
        return "formatted reply does not match BEGIN_META/BEGIN_FINAL schema"
    if expected_action is not None and parse_action(text) != expected_action:
        return f"action mismatch expected {expected_action!r} got {parse_action(text)!r}"
    thought = match.group("thought").strip()
    final = match.group("final").strip()
    for marker in ("【锚定】", "【分析】", "留联分层", "用户状态", "生理层", "心理层", "战术层", "【检验】", "【策略】"):
        if marker not in thought:
            return f"missing thought marker {marker}"
    if "fine_label=" not in thought:
        return "missing fine_label"
    if any(token in thought for token in ("改写", "蒸馏", "Claude", "模型", "数据集", "JSON")):
        return "thought contains processing trace"
    slots = parse_meta_slots(text)
    for key in REQUIRED_SLOT_KEYS:
        if key not in slots:
            return f"missing slot {key}"
        value = slots[key]
        if value in {"", "0", "1"}:
            return f"bad slot value {key}"
        if "<sep>" in value:
            return f"slot {key} contains sep"
    if not final:
        return "empty final"
    if any(trace in final for trace in ("改写", "蒸馏", "Claude", "模型", "数据集", "JSON")):
        return "final contains processing trace"
    if not CONTACT_WORD_RE.search(final):
        return "final lacks contact guidance"
    return None

def cache_path(args: argparse.Namespace, row: dict[str, Any], line_no: int) -> Path:
    key = str(row.get("index") or row.get("extra_info", {}).get("sample_id") or line_no)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return Path(args.cache_dir) / f"{safe}.json"


def load_cache(args: argparse.Namespace, row: dict[str, Any], line_no: int) -> dict[str, Any] | None:
    if args.no_cache:
        return None
    path = cache_path(args, row, line_no)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        expected_action = expected_action_from_row(row)
        if isinstance(cached, dict) and isinstance(cached.get("formatted"), str) and validate_formatted_reply(cached["formatted"], expected_action) is None:
            return cached
    except Exception:
        return None
    return None


def save_cache(args: argparse.Namespace, row: dict[str, Any], line_no: int, data: dict[str, Any]) -> None:
    if args.no_cache:
        return
    path = cache_path(args, row, line_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with cache_lock:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)


def save_raw(path: str, line_no: int, requested_model: str, returned_model: str, raw: str) -> None:
    with log_lock:
        ensure_parent(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- line {line_no} ---\n")
            f.write(f"requested_model={requested_model}\n")
            f.write(f"returned_model={returned_model}\n")
            f.write(raw.strip() + "\n\n")


def history_until_last_user(prompt: list[Any]) -> list[Any]:
    messages = prompt[1:] if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system" else prompt
    last_user_idx = None
    for idx, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_idx = idx
    if last_user_idx is None:
        return []
    return messages[: last_user_idx + 1]


def build_payload(row: dict[str, Any], line_no: int, previous_error: str | None = None) -> dict[str, Any]:
    prompt = row.get("prompt") or []
    system_prompt = prompt[0].get("content", "") if prompt and isinstance(prompt[0], dict) else ""
    history = history_until_last_user(prompt)
    payload: dict[str, Any] = {
        "source_line": line_no,
        "sample_id": row.get("index") or row.get("extra_info", {}).get("sample_id"),
        "system_prompt": system_prompt,
        "conversation_history": history,
        "current_user_input": next((str(m.get("content", "")) for m in reversed(history) if isinstance(m, dict) and m.get("role") == "user"), ""),
        "expected_action": expected_action_from_row(row),
        "extra_info_brief": {
            "turn_round": row.get("extra_info", {}).get("turn_round"),
            "rule_contact_round": row.get("extra_info", {}).get("rule_contact_round"),
            "question": row.get("extra_info", {}).get("question"),
        },
    }
    if previous_error:
        payload["previous_validation_error"] = previous_error
        payload["repair_instruction"] = "上一版输出未通过脚本校验。请只按指定 BEGIN_META/BEGIN_FINAL 格式重新输出完整回复，不要解释。"
    return payload


def call_claude(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[str, str]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_client(args).chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return response.choices[0].message.content or "", str(getattr(response, "model", ""))
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER_SECONDS))
    raise RuntimeError(str(last_exc))


def role_to_llama_factory(role: str) -> str:
    if role == "user":
        return "human"
    if role == "assistant":
        return "gpt"
    return role


def to_llama_factory_item(row: dict[str, Any], formatted_ground_truth: str) -> dict[str, Any]:
    prompt = row.get("prompt") or []
    system = ""
    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        system = str(prompt[0].get("content", ""))
        messages = prompt[1:]
    else:
        messages = prompt

    conversations: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = role_to_llama_factory(str(message.get("role", "")))
        if role not in {"human", "gpt"}:
            continue
        conversations.append({"from": role, "value": str(message.get("content", ""))})

    conversations.append({"from": "gpt", "value": formatted_ground_truth})
    return {"conversations": conversations, "system": system}


def process_one(args: argparse.Namespace, line_no: int, row: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(row)
    cached = load_cache(args, row, line_no)
    if cached is not None:
        formatted_ground_truth = clean_formatted_reply(cached["formatted"])
    else:
        previous_error = None
        formatted_ground_truth = None
        expected_action = expected_action_from_row(row)
        for _ in range(args.repair_rounds + 1):
            raw, returned_model = call_claude(args, build_payload(row, line_no, previous_error))
            save_raw(args.raw_log, line_no, args.model, returned_model, raw)
            candidate = clean_formatted_reply(raw)
            previous_error = validate_formatted_reply(candidate, expected_action)
            if previous_error is None:
                formatted_ground_truth = candidate
                break
        if formatted_ground_truth is None:
            raise ValueError(previous_error or "unknown validation error")
        save_cache(args, row, line_no, {"formatted": formatted_ground_truth})
    extra = output.setdefault("extra_info", {})
    if isinstance(extra, dict):
        extra.setdefault("original_ground_truth_before_claude", row.get("ground_truth", ""))
        extra["claude_distill_model"] = args.model
        extra["claude_distill_source_line"] = line_no
        extra["claude_distill_prompt"] = "contact_age_directed_direct_format_v4_no_target"
    output["ground_truth"] = formatted_ground_truth
    reward_model = output.get("reward_model")
    if isinstance(reward_model, dict):
        reward_model.setdefault("original_ground_truth_before_claude", row.get("ground_truth", ""))
        reward_model["ground_truth"] = formatted_ground_truth
        reward_model["style"] = "contact_stage_age_directed_claude_distilled"
    if args.output_format == "llama_factory":
        return to_llama_factory_item(output, formatted_ground_truth)
    return output


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.offset, args.limit)
    ensure_parent(args.output)
    ensure_parent(args.raw_log)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"input={args.input}")
    print(f"selected={len(rows)} offset={args.offset} limit={args.limit} model={args.model}")

    results: list[dict[str, Any] | None] = [None] * len(rows)
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, args, line_no, row): idx
            for idx, (line_no, row) in enumerate(rows)
        }
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                line_no, row = rows[idx]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = None
                    failure = {
                        "line_no": line_no,
                        "sample_id": row.get("index") or row.get("extra_info", {}).get("sample_id"),
                        "error": str(exc),
                    }
                    failures.append(failure)
                    print(f"line {line_no} failed: {exc}")
                pbar.update(1)

    failed_count = len(failures)
    if failed_count:
        failed_output = args.failed_output or args.output + ".failed.jsonl"
        ensure_parent(failed_output)
        with open(failed_output, "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"wrote {failed_count} failures to {failed_output}")
        if not args.allow_partial:
            raise RuntimeError(f"{failed_count} rows failed; refusing to write dirty SFT output. Rerun with --allow-partial to write successful rows.")

    clean_results = [item for item in results if item is not None]
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if args.output_format == "llama_factory":
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
            f.write("\n")
        else:
            for item in clean_results:
                f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, args.output)
    print(f"wrote {len(clean_results)} rows to {args.output} format={args.output_format} failed={failed_count}")


if __name__ == "__main__":
    main()
