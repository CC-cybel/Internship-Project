#!/usr/bin/env python3
"""Clean fixable issues from rewritten hard SFT dual-channel data.

This script is intentionally conservative:
- mechanically repairs human round markers and harmless assistant-format defects;
- rejects structurally broken samples and samples with empty visible replies/thoughts.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
from typing import Any


DEFAULT_INPUT = (
    "/data/chengch/project/data_remake/outputs/sft_hard_rewrite_v2/"
    "hard_rewrite_v2_sft_score4_5_clean_dual_full.json"
)

SYSTEM_MARKER_ANY_RE = re.compile(r"【系统(?:数据|状态)：当前\s*第?\s*(\d+)\s*轮】")
ACTION_TAG_RE = re.compile(r"<action>.*?</action>", re.S)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)
THINK_TAG_RE = re.compile(r"</?think>")
META_FINAL_RE = re.compile(
    r"\ABEGIN_META\n(?P<meta>.*?)\nEND_META\nBEGIN_FINAL\n(?P<final>.*?)\nEND_FINAL\Z",
    re.S,
)

REQUIRED_META_DEFAULTS = {
    "action": "",
    "thought": None,
    "slot_age": "未知",
    "slot_gender": "未知",
    "slot_name": "暂无",
    "slot_phone": "未获取",
    "slot_wechat": "未获取",
    "slot_symptom": "未知",
    "slot_duration": "暂无",
    "slot_medical_history": "暂无",
    "slot_relationship": "未知",
    "slot_medical_awareness": "未知",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-clean", default=None)
    parser.add_argument("--output-rejected", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def load_items(path: str) -> tuple[Any, list[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data, data["items"]
    if isinstance(data, list):
        return data, data
    raise ValueError("input must be list or dict with items list")


def visible_human(value: str) -> str:
    text = SYSTEM_MARKER_ANY_RE.sub("", value)
    text = ACTION_TAG_RE.sub("", text)
    return text.replace("<sep>", "").strip()


def normalize_human(value: str, round_num: int) -> tuple[str, list[str]]:
    fixes: list[str] = []
    markers = SYSTEM_MARKER_ANY_RE.findall(value)
    if not markers:
        fixes.append("add_missing_human_system_marker")
    elif len(markers) > 1:
        fixes.append("collapse_multiple_human_system_markers")
    if markers and (len(markers) != 1 or int(markers[-1]) != round_num or not re.search(rf"【系统数据：当前第\s*{round_num}\s*轮】", value)):
        fixes.append("normalize_human_system_marker")
    body = SYSTEM_MARKER_ANY_RE.sub("", value).rstrip()
    return f"{body}\n【系统数据：当前第 {round_num} 轮】", fixes


def parse_meta_final(value: str) -> tuple[str | None, str | None, list[str]]:
    fixes: list[str] = []
    stripped = value.strip()
    match = META_FINAL_RE.match(stripped)
    if match:
        return match.group("meta"), match.group("final"), fixes

    names = ["BEGIN_META", "END_META", "BEGIN_FINAL", "END_FINAL"]
    positions = [stripped.find(name) for name in names]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        return None, None, fixes
    fixes.append("repair_meta_final_outer_whitespace_or_boundaries")
    meta = stripped[positions[0] + len("BEGIN_META"):positions[1]]
    final = stripped[positions[2] + len("BEGIN_FINAL"):positions[3]]
    return meta.strip("\n"), final.strip("\n"), fixes


def parse_meta(meta: str) -> tuple[collections.OrderedDict[str, str], list[str], list[str]]:
    fields: collections.OrderedDict[str, str] = collections.OrderedDict()
    bad_lines: list[str] = []
    fixes: list[str] = []
    for raw in meta.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            bad_lines.append(line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in fields:
            fixes.append(f"drop_duplicate_meta_field:{key}")
            continue
        fields[key] = value.strip()
    return fields, bad_lines, fixes


def clean_final(final: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    cleaned = final.strip()
    if SYSTEM_MARKER_ANY_RE.search(cleaned):
        fixes.append("remove_system_marker_from_final")
        cleaned = SYSTEM_MARKER_ANY_RE.sub("", cleaned)
    if ACTION_TAG_RE.search(cleaned):
        fixes.append("remove_action_tag_from_final")
        cleaned = ACTION_TAG_RE.sub("", cleaned)
    if THINK_BLOCK_RE.search(cleaned):
        fixes.append("remove_think_block_from_final")
        cleaned = THINK_BLOCK_RE.sub("", cleaned)
    if THINK_TAG_RE.search(cleaned):
        fixes.append("remove_think_tag_from_final")
        cleaned = THINK_TAG_RE.sub("", cleaned)
    return cleaned.replace("\n\n\n", "\n\n").strip(), fixes


def normalize_text_for_compare(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("<sep>", ""))


def clean_gpt(value: str, current_human: str) -> tuple[str | None, list[str], list[str]]:
    meta, final, fixes = parse_meta_final(value)
    if meta is None or final is None:
        return None, [], ["gpt_unrepairable_meta_final_format"]
    fields, bad_lines, meta_fixes = parse_meta(meta)
    fixes.extend(meta_fixes)
    if bad_lines:
        fixes.append("drop_meta_lines_without_equals")
    thought = fields.get("thought", "")
    if not thought.strip():
        return None, fixes, ["gpt_empty_thought_unrepairable"]

    for key, default in REQUIRED_META_DEFAULTS.items():
        if key not in fields:
            if default is None:
                return None, fixes, [f"gpt_missing_required_meta_field_unrepairable:{key}"]
            fields[key] = default
            fixes.append(f"fill_missing_meta_field:{key}")

    final_clean, final_fixes = clean_final(final)
    fixes.extend(final_fixes)
    if len(final_clean.replace("<sep>", "").strip()) < 2:
        return None, fixes, ["gpt_empty_final_unrepairable"]
    if normalize_text_for_compare(final_clean) == normalize_text_for_compare(current_human):
        return None, fixes, ["gpt_final_copies_user_unrepairable"]

    ordered_lines = []
    for key in REQUIRED_META_DEFAULTS:
        ordered_lines.append(f"{key}={fields.get(key, '')}")
    for key, value in fields.items():
        if key not in REQUIRED_META_DEFAULTS:
            ordered_lines.append(f"{key}={value}")
    cleaned_value = "BEGIN_META\n" + "\n".join(ordered_lines) + "\nEND_META\nBEGIN_FINAL\n" + final_clean + "\nEND_FINAL"
    return cleaned_value, fixes, []


def clean_item(item: Any) -> tuple[Any | None, list[str], list[str]]:
    fixes: list[str] = []
    reject: list[str] = []
    if not isinstance(item, dict):
        return None, fixes, ["item_not_dict"]
    conv = item.get("conversations")
    if not isinstance(conv, list):
        return None, fixes, ["conversations_not_list"]
    if len(conv) % 2:
        return None, fixes, ["odd_message_count_unrepairable"]

    cleaned = copy.deepcopy(item)
    new_conv = []
    round_num = 1
    current_human = ""
    for message_index, message in enumerate(conv):
        if not isinstance(message, dict):
            return None, fixes, [f"message_not_dict:{message_index}"]
        role = message.get("from")
        value = message.get("value")
        expected_role = "human" if message_index % 2 == 0 else "gpt"
        if role != expected_role:
            return None, fixes, [f"bad_alternation:{message_index}:expected={expected_role},got={role}"]
        if not isinstance(value, str):
            return None, fixes, [f"bad_value_type:{message_index}:{type(value).__name__}"]
        if not value.strip():
            return None, fixes, [f"empty_message_value:{message_index}"]

        new_message = copy.deepcopy(message)
        if role == "human":
            current_human = visible_human(value)
            if not current_human:
                return None, fixes, [f"human_empty_visible_content:{message_index}"]
            new_value, human_fixes = normalize_human(value, round_num)
            fixes.extend(human_fixes)
            new_message["value"] = new_value
            round_num += 1
        else:
            new_value, gpt_fixes, gpt_reject = clean_gpt(value, current_human)
            fixes.extend(gpt_fixes)
            if gpt_reject:
                return None, fixes, [f"{reason}@message_{message_index}" for reason in gpt_reject]
            new_message["value"] = new_value
        new_conv.append(new_message)
    cleaned["conversations"] = new_conv
    return cleaned, fixes, reject


def dump_like_original(path: str, original_data: Any, items: list[Any]) -> None:
    out = {"items": items} if isinstance(original_data, dict) else items
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    stem = os.path.splitext(args.input)[0]
    output_clean = args.output_clean or stem + ".cleaned_fixable.json"
    output_rejected = args.output_rejected or stem + ".rejected_unfixable.json"
    report_path = args.report or stem + ".cleaned_fixable_report.json"

    original_data, items = load_items(args.input)
    cleaned_items = []
    rejected_items = []
    fix_counts: collections.Counter[str] = collections.Counter()
    reject_counts: collections.Counter[str] = collections.Counter()

    for index, item in enumerate(items):
        cleaned, fixes, reject = clean_item(item)
        for fix in fixes:
            fix_counts[fix] += 1
        if reject:
            for reason in reject:
                reject_counts[reason] += 1
            rejected_items.append({"_original_index": index, "_reject_reasons": reject, "item": item})
        else:
            cleaned_items.append(cleaned)

    dump_like_original(output_clean, original_data, cleaned_items)
    dump_like_original(output_rejected, original_data, rejected_items)
    report = {
        "input": args.input,
        "output_clean": output_clean,
        "output_rejected": output_rejected,
        "total_items": len(items),
        "cleaned_items": len(cleaned_items),
        "rejected_items": len(rejected_items),
        "fix_counts": dict(fix_counts.most_common()),
        "reject_counts": dict(reject_counts.most_common()),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
