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
from hashlib import sha1
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from rewrite_dialogues1_v2 import API_KEY, BASE_URL, MODEL_NAME


DEFAULT_INPUT = "/data/chengch/project/data_remake/intermediate/lgbt_rule_aug_v3_rewrite_cleaned_s2_dual_llama.json"
DEFAULT_OUTPUT = "/data/chengch/project/data_remake/runs/last_turn_value_slots_test.json"
DEFAULT_CACHE_DIR = "/data/chengch/project/data_remake/cache/last_turn_value_slots_test"
DEFAULT_RAW_LOG = "/data/chengch/project/data_remake/logs/last_turn_value_slots_test.txt"

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
RETRY_JITTER_SECONDS = 0.3

META_RE = re.compile(r"BEGIN_META\n(?P<meta>.*?)\nEND_META\nBEGIN_FINAL\n(?P<final>.*?)\nEND_FINAL", re.S)
ACTION_RE = re.compile(r"^action=(.*)$", re.M)

PROMPT = """
你负责把医疗咨询对话中最后一轮 assistant 的回答改成新的单轮 SFT 格式。

只允许改写最后一轮 assistant。历史对话、所有 human 内容、system prompt 不得修改。
不要在 thought 中提到 SFT、改写、原回复、assistant 原回答、JSON 等加工痕迹。
【锚定】里的当前轮次必须以最后一条 human 中的【系统数据：当前第 N 轮】为准，不得自行加一轮。

任务：
1. 改写最后一轮 assistant 的 thought。
   thought 必须包含【锚定】【分析】【检验】【策略】。
   【分析】必须按固定顺序包含：留联分层、用户状态、生理层、心理层、战术层。
   留联分层格式固定为：
   user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...
   fine_label 必须等于 user_type-core_need-conversion_barrier-lead_strategy。

2. 把 slot_values 改成 value 制，不要使用 0/1。
   必须输出这些字段：
   age, gender, name, relationship, phone, wechat, symptom, duration, medical_history, medical_awareness
   未知或没有获取时按 system 的约定写：
   age=未知, gender=未知, name=暂无, relationship=未知, phone=未获取, wechat=未获取,
   symptom=未知, duration=暂无, medical_history=暂无, medical_awareness=未知。
   只能根据当前轮和历史信息填写，严禁使用未来信息，严禁虚构。

3. response 基于原最后一轮 BEGIN_FINAL 轻微修正即可，优先保持原意和上下文连贯。
   如果原 response 已经可用，可以原样保留。
   response 中可以使用 <sep>，但 thought 和 slot_values 中禁止出现 <sep>。

枚举限制：
user_type 只能从 [青少年本人, 成人本人, 家长代询, 子女代询, 伴侣代询, 其他家属代询, 朋友代询, 未知] 中选择。
如果历史没有年龄、学生、未成年、成人、父母子女伴侣朋友等明确身份信息，user_type 必须写 未知，不能臆测青少年本人或成人本人。
只有明确出现“替/帮/给”孩子、父母、伴侣、朋友等别人咨询时，才允许使用代询类 user_type；“家里人不知道”“家人不理解”“不敢告诉家人”不是代询，仍按本人困扰处理，年龄未知时 user_type=未知。
core_need 只能从 [病情判断, 治疗方案, 既往治疗不满, 用药安全, 就医路径, 就医决策顾虑, 情绪倾诉, 危机求助, 家属照护无力, 其他] 中选择。
conversion_barrier 只能从 [医学认知不足, 路径不清, 费用顾虑, 信任顾虑, 效果顾虑, 用药顾虑, 隐私病耻, 患者不配合, 家庭沟通失效, 时间紧迫, 情绪承载不足, 危机安全风险, 暂无明显障碍, 信息不足] 中选择。
lead_strategy 只能从 [低压保密留联, 危机安全回电, 专家评估留联, 家属指导留联, 二次方案评估, 用药风险核对, 到院路径预约, 费用透明解释, 正规资质背书, 情绪承接转评估, 科普判断转留联, 暂不留联继续问诊] 中选择。
用户状态只能从 [平静, 犹豫, 害怕, 不信任, 对抗, 急迫, 配合, 敷衍, 未知] 中选择。
用户状态必须直接写上述枚举值之一，例如“用户状态：害怕，...” 严禁写“痛苦”“困惑”等非枚举词。
生理层只能从 [皮质醇, 多巴胺, 内啡肽, 催产素, 血清素] 中选择并写 ↑ 或 ↓。
心理层只能从 [自主感, 胜任感, 归属感, 优越感, 安全感, 认知节能, 损失厌恶] 中选择。
战术层只能从 [医学定性, 提问引导, 示弱反衬, 降维打击, 认知重构, 权威借势, 利益诱惑] 中选择。

输出 JSON only：
{
  "thought": "【锚定】...【分析】留联分层：user_type=...；core_need=...；conversion_barrier=...；lead_strategy=...；fine_label=...。用户状态：...。生理层：...。心理层：...。战术层：...。【检验】...【策略】...",
  "slot_values": {
    "age": "...",
    "gender": "...",
    "name": "...",
    "relationship": "...",
    "phone": "...",
    "wechat": "...",
    "symptom": "...",
    "duration": "...",
    "medical_history": "...",
    "medical_awareness": "..."
  },
  "response": "..."
}
"""

REQUIRED_SLOTS = [
    "age",
    "gender",
    "name",
    "relationship",
    "phone",
    "wechat",
    "symptom",
    "duration",
    "medical_history",
    "medical_awareness",
]
USER_STATES = {"平静", "犹豫", "害怕", "不信任", "对抗", "急迫", "配合", "敷衍", "未知"}
USER_TYPES = {"青少年本人", "成人本人", "家长代询", "子女代询", "伴侣代询", "其他家属代询", "朋友代询", "未知"}
CORE_NEEDS = {"病情判断", "治疗方案", "既往治疗不满", "用药安全", "就医路径", "就医决策顾虑", "情绪倾诉", "危机求助", "家属照护无力", "其他"}
CONVERSION_BARRIERS = {"医学认知不足", "路径不清", "费用顾虑", "信任顾虑", "效果顾虑", "用药顾虑", "隐私病耻", "患者不配合", "家庭沟通失效", "时间紧迫", "情绪承载不足", "危机安全风险", "暂无明显障碍", "信息不足"}
LEAD_STRATEGIES = {"低压保密留联", "危机安全回电", "专家评估留联", "家属指导留联", "二次方案评估", "用药风险核对", "到院路径预约", "费用透明解释", "正规资质背书", "情绪承接转评估", "科普判断转留联", "暂不留联继续问诊"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--raw-log", default=DEFAULT_RAW_LOG)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8, help="并发 API 请求数。")
    parser.add_argument("--submit-buffer", type=int, default=200, help="最多同时提交到线程池的任务数，避免全量数据一次性创建过多 future。")
    parser.add_argument("--model", default=os.environ.get("DASHSCOPE_MODEL_NAME", MODEL_NAME))
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", API_KEY))
    return parser.parse_args()


def ensure_parent(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def output_path_for_source(output_arg, input_arg, source_file):
    input_path = Path(input_arg)
    output_path = Path(output_arg)
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / source_file
    ensure_parent(output_arg)
    return output_path


def load_items(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()], "jsonl"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return (data.get("items", []) if isinstance(data, dict) else data), "json"


def load_entries(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        entries = []
        files = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
        for file_path in files:
            items, _ = load_items(file_path)
            if not isinstance(items, list):
                raise ValueError(f"{file_path} 不是 list 或包含 items 的 dict")
            for source_index, item in enumerate(items):
                entries.append({
                    "item": item,
                    "source_file": file_path.name,
                    "source_index": source_index,
                })
        return entries, [path.name for path in files]

    items, _ = load_items(input_path)
    if not isinstance(items, list):
        raise ValueError(f"{input_path} 不是 list 或包含 items 的 dict")
    return [
        {"item": item, "source_file": input_path.name, "source_index": i}
        for i, item in enumerate(items)
    ], [input_path.name]


def item_key(item, index, source_file=None, source_index=None):
    payload = {
        "index": index,
        "source_file": source_file or "",
        "source_index": source_index if source_index is not None else "",
        "system": item.get("system", ""),
        "conversations": item.get("conversations", []),
    }
    return sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def cache_path(cache_dir, key):
    return Path(cache_dir) / f"{key}.json"


def load_cache(cache_dir, key):
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("key") == key and data.get("status") == "ok":
        return data.get("item")
    return None


cache_lock = threading.Lock()
log_lock = threading.Lock()


def save_cache(cache_dir, key, payload):
    path = cache_path(cache_dir, key)
    tmp = path.with_suffix(".json.tmp")
    with cache_lock:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"key": key, **payload}, f, ensure_ascii=False)
        os.replace(tmp, path)


def save_raw(raw_log, index, raw):
    with log_lock:
        ensure_parent(raw_log)
        with open(raw_log, "a", encoding="utf-8") as f:
            f.write(f"--- ID: {index} ---\n{raw}\n\n")


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def find_last_gpt(conversations):
    for idx in range(len(conversations) - 1, -1, -1):
        if conversations[idx].get("from") == "gpt":
            return idx
    return None


def extract_action(old_value):
    match = ACTION_RE.search(old_value or "")
    return match.group(1).strip() if match else ""


def build_value(action, thought, slots, response):
    lines = ["BEGIN_META", f"action={action}", f"thought={thought}"]
    for key in [
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
    ]:
        lines.append(f"slot_{key}={slots.get(key, '')}")
    lines.extend(["END_META", "BEGIN_FINAL", response, "END_FINAL"])
    return "\n".join(lines)


def call_llm(client, model, payload):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER_SECONDS))
    raise last_exc


def validate_generated(data):
    if not isinstance(data, dict):
        return "response is not dict"
    thought = str(data.get("thought", ""))
    slots = data.get("slot_values")
    response = str(data.get("response", ""))
    if not thought or not isinstance(slots, dict) or not response:
        return "missing thought/slot_values/response"
    for marker in ["【锚定】", "【分析】", "留联分层：", "用户状态：", "生理层：", "心理层：", "战术层：", "【检验】", "【策略】"]:
        if marker not in thought:
            return f"missing marker {marker}"
    if any(token in thought for token in ["SFT", "改写", "原回复", "assistant", "JSON"]):
        return "thought contains processing trace"
    for key in REQUIRED_SLOTS:
        if key not in slots:
            return f"missing slot {key}"
        value = str(slots.get(key, ""))
        if value in {"0", "1"}:
            return f"slot {key} is still binary"
        if "<sep>" in value:
            return f"slot {key} contains <sep>"
    if "<sep>" in thought:
        return "thought contains <sep>"
    state_match = re.search(r"用户状态：([^，。；;,\s]+)", thought)
    if not state_match or state_match.group(1) not in USER_STATES:
        return "invalid 用户状态"
    seg_match = re.search(
        r"留联分层：user_type=([^；]+)；core_need=([^；]+)；conversion_barrier=([^；]+)；lead_strategy=([^；]+)；fine_label=([^。；]+)",
        thought,
    )
    if not seg_match:
        return "invalid 留联分层 format"
    user_type, core_need, barrier, strategy, fine_label = [part.strip() for part in seg_match.groups()]
    if user_type not in USER_TYPES:
        return "invalid user_type"
    if core_need not in CORE_NEEDS:
        return "invalid core_need"
    if barrier not in CONVERSION_BARRIERS:
        return "invalid conversion_barrier"
    if strategy not in LEAD_STRATEGIES:
        return "invalid lead_strategy"
    if fine_label != f"{user_type}-{core_need}-{barrier}-{strategy}":
        return "fine_label mismatch"
    return None


thread_local = threading.local()


def get_thread_client(args):
    client = getattr(thread_local, "client", None)
    if client is None:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        thread_local.client = client
    return client


def process_one(args, entry, index):
    item = entry["item"]
    key = item_key(item, index, entry.get("source_file"), entry.get("source_index"))
    cached = load_cache(args.cache_dir, key)
    if cached:
        return index, cached, None

    final_item = deepcopy(item)
    conversations = final_item.get("conversations", [])
    last_idx = find_last_gpt(conversations)
    if last_idx is None:
        return index, final_item, "no gpt turn"

    old_value = conversations[last_idx].get("value", "")
    match = META_RE.search(old_value)
    old_final = match.group("final").strip() if match else old_value
    action = extract_action(old_value)

    payload = {
        "system": final_item.get("system", ""),
        "history_before_last_assistant": conversations[:last_idx],
        "last_assistant_original": old_value,
        "last_response_original": old_final,
    }
    last_error = None
    raw = ""
    data = None
    for _ in range(2):
        if last_error:
            payload["previous_validation_error"] = last_error
            payload["repair_instruction"] = "上一次输出未通过校验。请严格修复该错误，只输出合法 JSON。"
        raw = call_llm(get_thread_client(args), args.model, payload)
        save_raw(args.raw_log, index, raw)
        data = parse_json(raw)
        last_error = validate_generated(data)
        if not last_error:
            break
    if last_error:
        raise ValueError(last_error)
    thought = str(data["thought"]).replace("<sep>", " ")
    slots = data["slot_values"]
    response = str(data["response"]).strip()
    conversations[last_idx]["value"] = build_value(action, thought, slots, response)
    save_cache(args.cache_dir, key, {"status": "ok", "item": final_item, "raw": raw})
    return index, final_item, None


def main():
    args = parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if Path(args.input).is_dir():
        Path(args.output).mkdir(parents=True, exist_ok=True)
    else:
        ensure_parent(args.output)
    ensure_parent(args.raw_log)

    entries, source_files = load_entries(args.input)
    selected = entries[args.offset: args.offset + args.limit if args.limit else None]
    results = [None] * len(selected)

    print(f"开始最后轮改写：input={args.input}, files={len(source_files)}, total={len(entries)}, offset={args.offset}, limit={len(selected)}, workers={args.max_workers}")
    if len(source_files) > 1:
        print("输入文件：" + ", ".join(source_files))
    pending = list(enumerate(selected))
    next_submit = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}

        def submit_more():
            nonlocal next_submit
            while next_submit < len(pending) and len(futures) < args.submit_buffer:
                local_i, entry = pending[next_submit]
                future = executor.submit(process_one, args, entry, args.offset + local_i)
                futures[future] = local_i
                next_submit += 1

        submit_more()
        with tqdm(total=len(pending)) as pbar:
            while futures:
                for future in as_completed(list(futures)):
                    local_idx = futures.pop(future)
                    try:
                        index, item, error = future.result()
                    except Exception as exc:
                        index = args.offset + local_idx
                        item = selected[local_idx]["item"]
                        error = str(exc)
                    results[local_idx] = item
                    if error:
                        print(f"ID {index} 失败: {error}")
                    pbar.update(1)
                    submit_more()
                    break

    grouped = {}
    for entry, item in zip(selected, results):
        source_file = entry["source_file"]
        grouped.setdefault(source_file, []).append(item)

    written = []
    for source_file, file_items in grouped.items():
        out_path = output_path_for_source(args.output, args.input, source_file)
        tmp = str(out_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(file_items, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, out_path)
        written.append(str(out_path))

    if len(written) == 1:
        print(f"完成：{written[0]}")
    else:
        print(f"完成：已分文件写入 {len(written)} 个文件到 {args.output}")


if __name__ == "__main__":
    main()
