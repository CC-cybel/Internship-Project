#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check LLaMA-Factory ShareGPT datasets used by last-turn value-slots SFT."""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_SLOT_KEYS = [
    "slot_age",
    "slot_gender",
    "slot_name",
    "slot_phone",
    "slot_wechat",
    "slot_symptom",
    "slot_duration",
    "slot_medical_history",
    "slot_relationship",
    "slot_medical_awareness",
]

ROUND_RE = re.compile(r"【系统数据：当前第\s*(\d+)\s*轮】")
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})(?!\d)")
META_RE = re.compile(r"BEGIN_META\s*\n(?P<meta>.*?)\nEND_META\s*\n+BEGIN_FINAL\s*\n(?P<final>.*?)\nEND_FINAL\s*$", re.S)
SLOT_LINE_RE = re.compile(r"^(slot_[a-z_]+)=(.*)$")


def load_dataset_info(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_items(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield line_no, json.loads(line)
                except Exception as exc:
                    yield line_no, {"__parse_error__": str(exc)}
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for idx, item in enumerate(data):
            yield idx, item
    else:
        yield 0, {"__top_level_not_list__": type(data).__name__}


def add_issue(issues, code, detail):
    issues.append({"code": code, "detail": detail})


def parse_target(value):
    if not isinstance(value, str):
        return None, None, "target_value_not_string"
    m = META_RE.match(value.strip())
    if not m:
        return None, None, "target_missing_meta_final_blocks"
    return m.group("meta"), m.group("final"), None


def parse_meta_slots(meta):
    slots = {}
    duplicate = []
    for raw_line in meta.splitlines():
        line = raw_line.strip()
        m = SLOT_LINE_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if key in slots:
            duplicate.append(key)
        slots[key] = value
    return slots, duplicate


def visible_user_text(value):
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<action>.*?</action>", "", value, flags=re.S)
    text = ROUND_RE.sub("", text)
    return text.replace("<sep>", "").strip()


def check_item(item, line_no):
    issues = []
    stats = Counter()

    if "__parse_error__" in item:
        add_issue(issues, "json_parse_error", item["__parse_error__"])
        return issues, stats
    if "__top_level_not_list__" in item:
        add_issue(issues, "top_level_not_list", item["__top_level_not_list__"])
        return issues, stats
    if not isinstance(item, dict):
        add_issue(issues, "item_not_dict", type(item).__name__)
        return issues, stats

    system = item.get("system")
    if not isinstance(system, str) or not system.strip():
        add_issue(issues, "system_missing_or_empty", "")
    elif "BEGIN_META" not in system or "Slot Schema" not in system:
        add_issue(issues, "system_prompt_schema_suspicious", system[:120].replace("\n", "\\n"))

    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        add_issue(issues, "conversations_not_list", type(conversations).__name__)
        return issues, stats
    if not conversations:
        add_issue(issues, "conversations_empty", "")
        return issues, stats
    if len(conversations) % 2 != 0:
        add_issue(issues, "odd_conversation_length", str(len(conversations)))

    expected = "human"
    gpt_indices = []
    round_nums = []
    for idx, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            add_issue(issues, "message_not_dict", f"msg_idx={idx}, type={type(msg).__name__}")
            continue
        role = msg.get("from")
        value = msg.get("value")
        if role not in {"human", "gpt"}:
            add_issue(issues, "bad_role", f"msg_idx={idx}, role={role!r}")
        if role != expected:
            add_issue(issues, "bad_alternation", f"msg_idx={idx}, expected={expected}, got={role!r}")
        expected = "gpt" if expected == "human" else "human"
        if not isinstance(value, str):
            add_issue(issues, "message_value_not_string", f"msg_idx={idx}, type={type(value).__name__}")
            continue
        if not value.strip():
            add_issue(issues, "message_value_empty", f"msg_idx={idx}, role={role}")
        if role == "human":
            if not visible_user_text(value):
                add_issue(issues, "human_visible_empty", f"msg_idx={idx}")
            nums = [int(x) for x in ROUND_RE.findall(value)]
            if nums:
                round_nums.extend(nums)
            if value.count("【系统数据：") > 1:
                add_issue(issues, "human_duplicate_system_round_marker", f"msg_idx={idx}")
        elif role == "gpt":
            gpt_indices.append(idx)
            if "【系统数据：" in value:
                add_issue(issues, "gpt_contains_system_round_marker", f"msg_idx={idx}")
            if "<action>" in value or "</action>" in value:
                add_issue(issues, "gpt_contains_action_tag", f"msg_idx={idx}")
            if value.strip().startswith("【人工客服】"):
                stats["history_manual客服_marker"] += 1
            if re.search(r"^slot_(?:age|gender|name|phone|wechat|symptom|duration|medical_history)=[01]\s*$", value, re.M):
                stats["history_binary_slot_marker"] += 1

    if round_nums:
        expected_rounds = list(range(1, len(round_nums) + 1))
        if round_nums != expected_rounds:
            add_issue(issues, "system_round_sequence_bad", f"rounds={round_nums[:20]}, expected_prefix={expected_rounds[:20]}")

    if not gpt_indices:
        add_issue(issues, "no_gpt_message", "")
        return issues, stats

    last_idx = gpt_indices[-1]
    last_value = conversations[last_idx].get("value") if isinstance(conversations[last_idx], dict) else None
    meta, final, err = parse_target(last_value)
    if err:
        add_issue(issues, err, f"last_gpt_msg_idx={last_idx}")
        return issues, stats

    slots, duplicate = parse_meta_slots(meta)
    if duplicate:
        add_issue(issues, "target_duplicate_slot_key", ",".join(sorted(set(duplicate))))
    for key in REQUIRED_SLOT_KEYS:
        if key not in slots:
            add_issue(issues, "target_missing_slot_key", key)
        elif slots[key] == "":
            add_issue(issues, "target_empty_slot_value", key)
        elif slots[key] in {"0", "1"}:
            add_issue(issues, "target_binary_slot_value", f"{key}={slots[key]}")

    if "thought=" not in meta:
        add_issue(issues, "target_missing_thought", "")
    if "留联分层" not in meta or "用户状态" not in meta:
        add_issue(issues, "target_thought_missing_user_model", "")
    final_clean = final.replace("<sep>", "").strip()
    if not final_clean:
        add_issue(issues, "target_final_empty", "")
    if "BEGIN_META" in final or "END_META" in final:
        add_issue(issues, "target_final_contains_meta_marker", "")
    if "【系统数据：" in final:
        add_issue(issues, "target_final_contains_system_round_marker", "")
    if "<action>" in final or "</action>" in final:
        add_issue(issues, "target_final_contains_action_tag", "")
    if final.count("?") + final.count("？") > 3:
        add_issue(issues, "target_final_too_many_questions", str(final.count("?") + final.count("？")))
    if len(final_clean) < 2:
        add_issue(issues, "target_final_too_short", final_clean)

    if PHONE_RE.search(final):
        stats["target_final_contains_phone_number"] += 1

    return issues, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset", required=True, help="Comma separated dataset names, same as yaml.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sample-per-issue", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info = load_dataset_info(dataset_dir / "dataset_info.json")
    names = [x.strip() for x in args.dataset.split(",") if x.strip()]

    summary = {
        "dataset_dir": str(dataset_dir),
        "datasets": {},
        "total_items": 0,
        "total_items_with_issue": 0,
        "issue_counts": Counter(),
        "risk_stats": Counter(),
    }
    examples = defaultdict(list)
    issue_path = Path(args.output_prefix + ".issues.jsonl")
    report_path = Path(args.output_prefix + ".report.json")

    severe_codes = {
        "json_parse_error",
        "top_level_not_list",
        "item_not_dict",
        "system_missing_or_empty",
        "conversations_not_list",
        "conversations_empty",
        "message_not_dict",
        "bad_role",
        "bad_alternation",
        "message_value_not_string",
        "message_value_empty",
        "human_visible_empty",
        "no_gpt_message",
        "target_value_not_string",
        "target_missing_meta_final_blocks",
        "target_missing_slot_key",
        "target_empty_slot_value",
        "target_binary_slot_value",
        "target_final_empty",
        "target_final_contains_meta_marker",
        "target_final_contains_system_round_marker",
    }

    with issue_path.open("w", encoding="utf-8") as out:
        for name in names:
            entry = info.get(name)
            if not entry:
                raise KeyError(f"dataset {name!r} not found in dataset_info.json")
            file_path = dataset_dir / entry["file_name"]
            ds_counter = Counter()
            ds_risk = Counter()
            total = 0
            issue_items = 0
            severe_items = 0
            for line_no, item in iter_items(file_path):
                total += 1
                issues, stats = check_item(item, line_no)
                ds_risk.update(stats)
                if issues:
                    issue_items += 1
                    codes = sorted({x["code"] for x in issues})
                    if any(code in severe_codes for code in codes):
                        severe_items += 1
                    for issue in issues:
                        ds_counter[issue["code"]] += 1
                        if len(examples[issue["code"]]) < args.sample_per_issue:
                            examples[issue["code"]].append({
                                "dataset": name,
                                "file": entry["file_name"],
                                "line_or_index": line_no,
                                "detail": issue["detail"],
                            })
                    out.write(json.dumps({
                        "dataset": name,
                        "file": entry["file_name"],
                        "line_or_index": line_no,
                        "issues": issues,
                    }, ensure_ascii=False) + "\n")
            summary["datasets"][name] = {
                "file": entry["file_name"],
                "total_items": total,
                "items_with_issue": issue_items,
                "severe_items": severe_items,
                "issue_counts": dict(ds_counter),
                "risk_stats": dict(ds_risk),
            }
            summary["total_items"] += total
            summary["total_items_with_issue"] += issue_items
            summary["issue_counts"].update(ds_counter)
            summary["risk_stats"].update(ds_risk)

    serializable = dict(summary)
    serializable["issue_counts"] = dict(summary["issue_counts"])
    serializable["risk_stats"] = dict(summary["risk_stats"])
    serializable["examples"] = dict(examples)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "report": str(report_path),
        "issues": str(issue_path),
        "total_items": summary["total_items"],
        "total_items_with_issue": summary["total_items_with_issue"],
        "top_issues": summary["issue_counts"].most_common(20),
        "risk_stats": summary["risk_stats"].most_common(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
