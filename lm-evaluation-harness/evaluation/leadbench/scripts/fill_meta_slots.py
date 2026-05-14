#!/usr/bin/env python3
"""Fill BEGIN_META blocks for LeadBench golden-history assistant messages.

The script keeps the original JSONL schema and rewrites assistant `content`
from:

    原始回复

to:

    BEGIN_META
    action=...
    thought=...
    slot_age=...
    ...
    END_META
    BEGIN_FINAL
    原始回复
    END_FINAL

Slot values are extracted from the conversation history by an OpenAI-compatible
judge model. Missing values are kept empty instead of converted to 0/1.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error, request


DEFAULT_INPUT = (
    "/data/chengch/lm-evaluation-harness/evaluation/leadbench/"
    "data/dataset/psychiatry/golden_history_input_v1.jsonl"
)
DEFAULT_OUTPUT = (
    "/data/chengch/lm-evaluation-harness/evaluation/leadbench/"
    "data/dataset/psychiatry/golden_history_input_v1.with_meta.jsonl"
)

SLOT_KEYS = [
    "age",
    "gender",
    "name",
    "phone",
    "wechat",
    "symptom",
    "duration",
    "medical_history",
    "relationship",
]


SYSTEM_PROMPT = """你是医疗咨询对话的结构化标注员。请根据截至当前助手回复之前的完整对话历史，抽取已经由用户明确提供或强烈可推断的信息。

要求：
- 只输出 JSON，不要 Markdown，不要解释。
- 不要使用 0/1。槽位填真实值；未知填空字符串。
- age 只填数字，如 28。
- gender 只填 男 或 女；未知填空字符串。
- relationship 填咨询者与患者关系，如 本人、妹妹、家人、妻子、朋友。
- symptom 填主诉症状，尽量简短但保留关键信息。
- duration 填病程时长原文或归一化短语。
- medical_history 填既往史、检查史、用药史、诱因等已知信息。
- phone、wechat 只在用户明确提供联系方式时填写。
- action：若最后一条用户输入包含 <action>...</action>，填其中动作文本；否则为空字符串。
- thought：用一句中文简述当前回复的策略和已知槽位依据，不能换行。

JSON 字段必须包含：
action, thought, age, gender, name, phone, wechat, symptom, duration, medical_history, relationship
"""


USER_PROMPT_TEMPLATE = """请抽取当前助手回复前已经知道的槽位。

对话历史：
{history}

当前助手回复原文：
{assistant_content}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a judge model to add real-value BEGIN_META blocks to assistant messages."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input atomically. A .bak file is written unless --no-backup is set.",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not write a .bak file with --in-place.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of parallel API calls.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing --output prefix and continue.")
    parser.add_argument("--dry-run", action="store_true", help="Print first converted sample without writing.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not rewrite assistant messages that already contain BEGIN_META.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Use simple local regex extraction when the judge API fails. Default is to stop on API errors.",
    )
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL_NAME", "qwen-max"))
    parser.add_argument("--api-base", default=os.getenv("JUDGE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY"))
    parser.add_argument(
        "--enable-thinking",
        default=os.getenv("JUDGE_ENABLE_THINKING", "true"),
        choices=["true", "false", "True", "False", "1", "0"],
        help="Passed through to DashScope-compatible extra_body when supported.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def already_wrapped(text: str) -> bool:
    return "BEGIN_META" in text and "BEGIN_FINAL" in text and "END_FINAL" in text


def strip_wrapper(text: str) -> str:
    if not already_wrapped(text):
        return text
    match = re.search(r"BEGIN_FINAL\s*(.*?)\s*END_FINAL\s*$", text, flags=re.S)
    return match.group(1).strip() if match else text


def action_from_last_user(messages: List[Dict[str, Any]], assistant_idx: int) -> str:
    for msg in reversed(messages[:assistant_idx]):
        if msg.get("role") == "user":
            match = re.search(r"<action>(.*?)</action>", msg.get("content", ""), flags=re.S)
            return match.group(1).strip() if match else ""
    return ""


def format_history(messages: List[Dict[str, Any]], assistant_idx: int) -> str:
    lines = []
    for msg in messages[:assistant_idx]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if enable_thinking:
        # Match the repo's APIModel behavior: OpenAI extra_body is merged into
        # the HTTP body, so raw urllib requests should put these keys top-level.
        payload["chat_template_kwargs"] = {"thinking": True, "enable_thinking": True}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def preflight_network(args: argparse.Namespace) -> None:
    host = re.sub(r"^https?://", "", args.api_base).split("/", 1)[0].split(":", 1)[0]
    try:
        addrs = socket.getaddrinfo(host, 443)
    except socket.gaierror as exc:
        proxy_hint = (
            f" HTTP_PROXY={os.getenv('HTTP_PROXY', '')}"
            f" HTTPS_PROXY={os.getenv('HTTPS_PROXY', '')}"
            f" http_proxy={os.getenv('http_proxy', '')}"
            f" https_proxy={os.getenv('https_proxy', '')}"
        )
        raise RuntimeError(
            f"Cannot resolve API host {host}: {exc}.{proxy_hint} "
            "Check DNS or proxy settings, and make sure proxy URLs use ASCII ':' not '：'."
        ) from exc
    resolved = sorted({item[4][0] for item in addrs})
    print(f"Resolved {host}: {', '.join(resolved[:4])}")


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_slots(raw: Dict[str, Any], fallback_action: str) -> Dict[str, str]:
    result = {"action": str(raw.get("action") or fallback_action or "").strip()}
    thought = str(raw.get("thought") or "根据当前对话历史补全已知槽位，并保持原助手回复不变。")
    result["thought"] = " ".join(thought.split())
    for key in SLOT_KEYS:
        value = raw.get(key, "")
        if value is None:
            value = ""
        if isinstance(value, list):
            value = "、".join(str(x).strip() for x in value if str(x).strip())
        result[key] = " ".join(str(value).split())
    if result["gender"] not in {"", "男", "女"}:
        if "女" in result["gender"]:
            result["gender"] = "女"
        elif "男" in result["gender"]:
            result["gender"] = "男"
    age_match = re.search(r"\d+", result["age"])
    if age_match:
        result["age"] = age_match.group(0)
    return result


def fallback_slots(messages: List[Dict[str, Any]], assistant_idx: int) -> Dict[str, str]:
    text = "\n".join(msg.get("content", "") for msg in messages[:assistant_idx] if msg.get("role") == "user")
    slots = {key: "" for key in SLOT_KEYS}
    action = action_from_last_user(messages, assistant_idx)
    age = re.search(r"(\d{1,3})\s*岁", text)
    if age:
        slots["age"] = age.group(1)
    if re.search(r"女生|女性|女的|妹妹|老婆|妻子|妈妈|母亲|女儿|姐姐", text):
        slots["gender"] = "女"
    elif re.search(r"男生|男性|男的|老公|丈夫|爸爸|父亲|儿子|哥哥", text):
        slots["gender"] = "男"
    phone = re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text)
    if phone:
        slots["phone"] = phone.group(0)
    if re.search(r"我自己|本人|自己的情况", text):
        slots["relationship"] = "本人"
    elif "妹妹" in text:
        slots["relationship"] = "妹妹"
    elif re.search(r"老婆|妻子", text):
        slots["relationship"] = "妻子"
    elif re.search(r"家人|家属", text):
        slots["relationship"] = "家人"
    slots["thought"] = "API 抽取失败，使用本地规则补全可确定槽位。"
    slots["action"] = action
    return slots


def build_meta(slots: Dict[str, str]) -> str:
    lines = [
        "BEGIN_META",
        f"action={slots.get('action', '')}",
        f"thought={slots.get('thought', '')}",
    ]
    for key in SLOT_KEYS:
        lines.append(f"slot_{key}={slots.get(key, '')}")
    lines.append("END_META")
    return "\n".join(lines)


def wrap_content(content: str, slots: Dict[str, str]) -> str:
    final = strip_wrapper(content).strip()
    return f"{build_meta(slots)}\nBEGIN_FINAL\n{final}\nEND_FINAL"


def call_with_retries(args: argparse.Namespace, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not args.api_key:
        raise RuntimeError("JUDGE_API_KEY is not set")

    last_exc: Optional[BaseException] = None
    for attempt in range(args.max_retries):
        try:
            content = chat_completion(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                enable_thinking=args.enable_thinking.lower() in {"true", "1"},
            )
            return extract_json_object(content)
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"judge API failed after {args.max_retries} retries: {last_exc}")


def process_assistant_turn(
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    assistant_idx: int,
) -> Tuple[int, str]:
    assistant_content = messages[assistant_idx].get("content", "")
    if args.skip_existing and already_wrapped(assistant_content):
        return assistant_idx, assistant_content

    fallback_action = action_from_last_user(messages, assistant_idx)
    prompt = USER_PROMPT_TEMPLATE.format(
        history=format_history(messages, assistant_idx),
        assistant_content=strip_wrapper(assistant_content),
    )
    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        raw_slots = call_with_retries(args, api_messages)
        slots = normalize_slots(raw_slots, fallback_action)
    except Exception as exc:
        if not args.allow_fallback:
            raise RuntimeError(
                f"assistant_idx={assistant_idx} judge API failed. "
                "Fix network/API settings or rerun with --allow-fallback."
            ) from exc
        print(f"[WARN] assistant_idx={assistant_idx} API failed, using fallback: {exc}", file=sys.stderr)
        slots = fallback_slots(messages, assistant_idx)
    return assistant_idx, wrap_content(assistant_content, slots)


def process_row(args: argparse.Namespace, row: Dict[str, Any]) -> Dict[str, Any]:
    new_row = copy.deepcopy(row)
    messages = new_row.get("messages", [])
    assistant_indices = [
        i for i, msg in enumerate(messages)
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str)
    ]
    if not assistant_indices:
        return new_row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(process_assistant_turn, args, messages, idx)
            for idx in assistant_indices
        ]
        for future in concurrent.futures.as_completed(futures):
            idx, content = future.result()
            messages[idx]["content"] = content
    return new_row


def completed_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.in_place:
        output_path = input_path.with_suffix(input_path.suffix + ".filled.tmp")

    if not args.api_key and not args.dry_run:
        raise SystemExit("JUDGE_API_KEY is required. Export it or pass --api-key.")
    if args.api_key:
        preflight_network(args)

    rows = load_jsonl(input_path, args.limit)
    start = completed_count(output_path) if args.resume and not args.in_place else 0
    if start:
        print(f"Resuming from record {start}; existing output: {output_path}")
    rows_to_process = rows[start:]

    if args.dry_run:
        sample = process_row(args, rows_to_process[0])
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for offset, row in enumerate(rows_to_process, start=start + 1):
            converted = process_row(args, row)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{offset}/{len(rows)}] wrote id={converted.get('id', converted.get('key', 'unknown'))}")

    if args.in_place:
        if not args.no_backup:
            backup_path = input_path.with_suffix(input_path.suffix + ".bak")
            input_path.replace(backup_path)
            print(f"Backup written to {backup_path}")
        output_path.replace(input_path)
        print(f"Updated input in place: {input_path}")
    else:
        print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
