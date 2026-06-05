#!/usr/bin/env python3
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

import httpx
from openai import OpenAI
from tqdm import tqdm

INPUT_FILE = "/data/chengch/project/data_remake/intermediate/opd_offline_full_v3_a2r8_train.jsonl"
OUTPUT_FILE = "/data/chengch/project/data_remake/outputs/claude_distlill.jsonl"
RAW_LOG = "/data/chengch/project/data_remake/logs/claude_distlill_raw.txt"
API_KEY = os.environ.get("CLAUDE_DISTILL_API_KEY", "")
BASE_URL = os.environ.get("CLAUDE_DISTILL_BASE_URL", "https://router-hk.dingningtalk.com/v1")
MODEL_NAME = os.environ.get("CLAUDE_DISTILL_MODEL_NAME", "anthropic/claude-sonnet-4.5")
REQUEST_TIMEOUT = float(os.environ.get("CLAUDE_DISTILL_TIMEOUT", "300"))

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_JITTER_SECONDS = 0.5

META_RE = re.compile(r"BEGIN_META\n(?P<meta>.*?)\nEND_META\nBEGIN_FINAL\n(?P<final>.*?)\nEND_FINAL", re.S)
ACTION_RE = re.compile(r"^action=(.*)$", re.M)
ROUND_RE = re.compile(r"当前第\s*(\d+)\s*轮|第\s*(\d+)\s*轮")

CONTACT_PATTERNS = [
    r"留(?:个|下|一下)?.{0,8}(?:电话|手机号|联系方式|微信)",
    r"(?:电话|手机号|联系方式|微信号|微信)多少",
    r"方便.{0,8}(?:电话|微信|手机号|联系方式|号码)",
    r"(?:电话|微信).{0,10}(?:沟通|联系|回访|回电|咨询|指导|评估)",
    r"(?:加|添加).{0,6}(?:微信|你)",
    r"(?:老师|医生).{0,10}(?:回电|联系你|联系您|加你|加您)",
    r"(?:稍后|晚点|一会).{0,10}(?:联系|回电|加)",
    r"(?:预约|名额|定位|路线|挂号).{0,12}(?:微信|电话|联系|发给|发送|预留)",
    r"(?:接听|通过一下|通过该号码|该号码联系)",
]
CONTACT_RE = re.compile("|".join(CONTACT_PATTERNS))

REQUIRED_SLOT_KEYS = [
    "age", "gender", "name", "phone", "wechat", "symptom", "duration",
    "medical_history", "relationship", "medical_awareness",
]

PROMPT = """
你是资深医疗咨询对话改写专家。你的任务是把给定对话的最后一轮 assistant 回复改写得更自然、更强转化、更专业，但必须严格遵守格式和事实边界。

只允许改写最后一轮 assistant。历史 assistant、所有 human、system prompt 不得修改。
不要使用任何未来信息，只能根据当前轮和历史信息填写 thought 与槽位。
不要在 thought 中提到“改写、原回复、JSON、模型、Claude、SFT”等加工痕迹。

改写目标：
1. 当前最后一轮属于套联/留联阶段，要围绕电话或微信转化，把理由讲得更可信、更有必要性。
2. 用户可见回复要自然、有压迫但不过度，不能像客服话术，不能列表化。
3. 保留或增强原 system prompt 中的留联策略、拒绝处理、询问密度、语言风格等要求。
4. 如果用户已经提供联系方式，则本轮重点应确认、安抚、说明后续联系；如果尚未提供，则本轮应明确引导留下电话或微信。

输出必须是 JSON object，字段如下：
必须返回严格合法 JSON，不要包裹 ```json 代码块；字符串内部严禁使用未转义的半角双引号，如需强调请改用中文表述或中文引号。
{
  "action": "从原最后一轮 BEGIN_META 的 action 继承，没有则空字符串",
  "thought": "【锚定】...【分析】留联分层：user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。用户状态：...。生理层：...。心理层：...。战术层：...。【检验】...【策略】...",
  "slot_values": {
    "age": "未知或具体值",
    "gender": "男/女/未知",
    "name": "暂无或具体称呼",
    "phone": "未获取或具体号码",
    "wechat": "未获取或具体微信号",
    "symptom": "未知或主诉症状",
    "duration": "暂无或具体时长",
    "medical_history": "暂无或既往史/诊断史/用药史/检查史",
    "relationship": "本人/母亲/父亲/伴侣/子女/朋友/其他家属/未知",
    "medical_awareness": "未知/小白/半懂/专业/误区明显"
  },
  "response": "BEGIN_FINAL 中用户可见回复"
}

硬性格式要求：
- thought 必须包含【锚定】【分析】【检验】【策略】。
- 【分析】内部字段顺序固定为：留联分层、用户状态、生理层、心理层、战术层。
- 留联分层必须包含 user_type/core_need/conversion_barrier/lead_strategy/fine_label，fine_label 必须等于四项用短横线拼接。
- slot_values 必须使用 value 制，严禁 0/1。
- thought 和 slot_values 中严禁出现 <sep>。
- response 可以使用 <sep>，但不要滥用。
- 最终回复不得使用 Markdown，不得输出 BEGIN_META/BEGIN_FINAL，脚本会负责拼接。

枚举：
user_type: [青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知]
core_need: [病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他]
conversion_barrier: [医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足]
lead_strategy: [低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊]
用户状态: [平静, 犹豫, 害怕, 不信任, 对抗, 急迫, 配合, 敷衍, 未知]
生理层: [皮质醇, 多巴胺, 内啡肽, 催产素, 血清素]，必须写 ↑ 或 ↓
心理层: [自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶]
战术层: [医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑]
"""

thread_local = threading.local()
log_lock = threading.Lock()


def get_client(args):
    client = getattr(thread_local, "client", None)
    if client is None:
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            http_client=httpx.Client(trust_env=False, timeout=args.timeout),
        )
        thread_local.client = client
    return client


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Select contact-stage last turns and rewrite them with Claude.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--raw-log", default=RAW_LOG)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-scan", type=int, default=0, help="最多扫描多少行，0 表示全量扫描。")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def find_last_gpt(conversations):
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("from") == "gpt":
            return i
    return None


def split_assistant_value(value):
    match = META_RE.search(value or "")
    if not match:
        return "", value or ""
    meta = match.group("meta")
    final = match.group("final").strip()
    action_match = ACTION_RE.search(meta)
    action = action_match.group(1).strip() if action_match else ""
    return action, final


def is_contact_stage(item):
    conversations = item.get("conversations") or []
    last_idx = find_last_gpt(conversations)
    if last_idx is None:
        return False
    value = conversations[last_idx].get("value", "")
    action, final = split_assistant_value(value)
    text = f"{action}\n{final}\n{value}"
    if any(word in action for word in ["留", "套联", "联系方式", "电话", "微信"]):
        return True
    return CONTACT_RE.search(text) is not None


def iter_selected(path, offset, limit, max_scan):
    selected = []
    matched = 0
    scanned = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_scan and scanned >= max_scan:
                break
            scanned += 1
            if not line.strip():
                continue
            item = json.loads(line)
            if not is_contact_stage(item):
                continue
            if matched >= offset and (not limit or len(selected) < limit):
                selected.append((line_no, item))
            matched += 1
            if limit and len(selected) >= limit:
                break
    return selected, matched, scanned


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def validate(data):
    if not isinstance(data, dict):
        return "not dict"
    for key in ["action", "thought", "slot_values", "response"]:
        if key not in data:
            return f"missing {key}"
    thought = str(data.get("thought", ""))
    slots = data.get("slot_values")
    response = str(data.get("response", ""))
    if not isinstance(slots, dict):
        return "slot_values not dict"
    for marker in ["【锚定】", "【分析】", "留联分层：", "用户状态：", "生理层：", "心理层：", "战术层：", "【检验】", "【策略】"]:
        if marker not in thought:
            return f"missing {marker}"
    if any(token in thought for token in ["改写", "原回复", "JSON", "Claude", "SFT"]):
        return "thought contains processing trace"
    if "<sep>" in thought:
        return "thought contains sep"
    for key in REQUIRED_SLOT_KEYS:
        if key not in slots:
            return f"missing slot {key}"
        value = str(slots.get(key, ""))
        if value in {"0", "1"}:
            return f"binary slot {key}"
        if "<sep>" in value:
            return f"slot {key} contains sep"
    if not response.strip():
        return "empty response"
    return None


def build_value(action, thought, slots, response):
    lines = ["BEGIN_META", f"action={action}", f"thought={thought}"]
    for key in REQUIRED_SLOT_KEYS:
        lines.append(f"slot_{key}={slots.get(key, '')}")
    lines.extend(["END_META", "BEGIN_FINAL", response.strip(), "END_FINAL"])
    return "\n".join(lines)


def save_raw(path, line_no, raw):
    with log_lock:
        ensure_parent(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- line {line_no} ---\n{raw}\n\n")


def call_claude(args, payload):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_client(args).chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=args.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER_SECONDS))
    raise last_exc


def process_one(args, line_no, item):
    original = deepcopy(item)
    conversations = original.get("conversations") or []
    last_idx = find_last_gpt(conversations)
    old_value = conversations[last_idx].get("value", "")
    old_action, old_final = split_assistant_value(old_value)
    payload = {
        "source_line": line_no,
        "system": original.get("system", ""),
        "history_before_last_assistant": conversations[:last_idx],
        "last_assistant_original": old_value,
        "last_action_original": old_action,
        "last_final_original": old_final,
    }
    last_error = None
    raw = ""
    data = None
    for _ in range(2):
        if last_error:
            payload["previous_validation_error"] = last_error
            payload["repair_instruction"] = "上一版输出未通过校验，请只修复格式和值域问题，输出合法 JSON object。"
        raw = call_claude(args, payload)
        save_raw(args.raw_log, line_no, raw)
        try:
            data = parse_json(raw)
        except Exception as exc:
            last_error = f"json parse error: {exc}"
            continue
        last_error = validate(data)
        if not last_error:
            break
    if last_error:
        raise ValueError(last_error)
    action = str(data.get("action", old_action)).strip()
    conversations[last_idx]["value"] = build_value(
        action,
        str(data["thought"]).replace("<sep>", " ").strip(),
        data["slot_values"],
        str(data["response"]).strip(),
    )
    original["claude_distill_source_line"] = line_no
    return original


def main():
    args = parse_args()
    ensure_parent(args.output)
    ensure_parent(args.raw_log)
    selected, matched, scanned = iter_selected(args.input, args.offset, args.limit, args.max_scan)
    print(f"scanned={scanned} matched_contact_stage={matched} selected={len(selected)} offset={args.offset} limit={args.limit}")
    results = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, args, line_no, item): i
            for i, (line_no, item) in enumerate(selected)
        }
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                i = futures[future]
                line_no = selected[i][0]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    print(f"line {line_no} failed: {exc}")
                    results[i] = selected[i][1]
                    results[i]["claude_distill_error"] = str(exc)
                    results[i]["claude_distill_source_line"] = line_no
                pbar.update(1)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, args.output)
    print(f"wrote {len(results)} items to {args.output}")


if __name__ == "__main__":
    main()
