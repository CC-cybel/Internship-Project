#!/usr/bin/env python3
"""Quality checks for hard_rewrite_v2.json object-style rewrite outputs.

Expected assistant message value:
{
  "thought": "...",
  "slot_values": {
    "age": "...",
    ...
  },
  "response": "..."
}

The script writes:
  - *.object_quality_report.md
  - *.object_quality_issues.jsonl
  - *.object_severe_bad_item_indices.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


DEFAULT_INPUT = (
    "/data/chengch/project/data_remake/outputs/raw_hard_rewrite_v2/"
    "hard_rewrite_v2.json"
)

SYSTEM_MARKER_ANY_RE = re.compile(r"【系统(?:数据|状态)：当前\s*第?\s*(\d+)\s*轮】")
SYSTEM_MARKER_STRICT_RE = re.compile(r"【系统数据：当前第\s*(\d+)\s*轮】")
ACTION_TAG_RE = re.compile(r"<action>.*?</action>", re.S)
THINK_TAG_RE = re.compile(r"</?think>")

REQUIRED_GPT_KEYS = ["thought", "slot_values", "response"]
REQUIRED_SLOT_KEYS = [
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

SEVERE_ISSUES = {
    "item_not_dict",
    "conversations_missing",
    "conversations_not_list",
    "message_not_dict",
    "bad_from_field",
    "bad_value_type",
    "empty_message_value",
    "odd_message_count",
    "bad_alternation",
    "same_role_consecutive",
    "human_empty_visible_content",
    "human_multiple_system_markers",
    "gpt_value_not_dict",
    "gpt_missing_required_key",
    "gpt_bad_thought_type",
    "gpt_empty_thought",
    "gpt_bad_response_type",
    "gpt_empty_response",
    "gpt_response_contains_system_marker",
    "gpt_response_contains_action_tag",
    "gpt_response_contains_think_tag",
    "gpt_response_contains_json_fragment",
    "gpt_response_copies_current_user",
    "gpt_slot_values_not_dict",
    "gpt_missing_slot_key",
    "gpt_bad_slot_value_type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check object-style hard_rewrite_v2 JSON quality."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON file.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to input path without .json.",
    )
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--min-response-chars", type=int, default=2)
    return parser.parse_args()


def load_items(path: str) -> list[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Input must be a list or an object with list field 'items'.")


def preview(value: Any, limit: int = 500) -> str:
    if isinstance(value, str):
        return value[:limit]
    return repr(value)[:limit]


class IssueCollector:
    def __init__(self, sample_limit: int) -> None:
        self.counts: collections.Counter[str] = collections.Counter()
        self.items: dict[str, set[int]] = collections.defaultdict(set)
        self.samples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        self.records: list[dict[str, Any]] = []
        self.sample_limit = sample_limit

    def add(
        self,
        issue: str,
        item_index: int,
        message_index: int | None = None,
        detail: str = "",
        message: Any = None,
    ) -> None:
        self.counts[issue] += 1
        self.items[issue].add(item_index)
        record: dict[str, Any] = {"issue": issue, "item_index": item_index}
        if message_index is not None:
            record["message_index"] = message_index
        if detail:
            record["detail"] = detail
        if isinstance(message, dict):
            record["from"] = message.get("from")
            record["value_preview"] = preview(message.get("value"))
        elif message is not None:
            record["value_preview"] = preview(message)
        self.records.append(record)
        if len(self.samples[issue]) < self.sample_limit:
            self.samples[issue].append(record)


def visible_human_text(value: str) -> str:
    text = SYSTEM_MARKER_ANY_RE.sub("", value)
    text = ACTION_TAG_RE.sub("", text)
    return text.replace("<sep>", "").strip()


def normalize_visible_text(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("<sep>", ""))


def check_gpt_value(
    value: Any,
    item_index: int,
    message_index: int,
    current_user_text: str,
    collector: IssueCollector,
    min_response_chars: int,
    message: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        collector.add(
            "gpt_value_not_dict",
            item_index,
            message_index,
            detail=type(value).__name__,
            message=message,
        )
        return

    for key in REQUIRED_GPT_KEYS:
        if key not in value:
            collector.add("gpt_missing_required_key", item_index, message_index, key, message)

    thought = value.get("thought")
    if not isinstance(thought, str):
        collector.add(
            "gpt_bad_thought_type",
            item_index,
            message_index,
            detail=type(thought).__name__,
            message=message,
        )
    elif not thought.strip():
        collector.add("gpt_empty_thought", item_index, message_index, message=message)

    response = value.get("response")
    if not isinstance(response, str):
        collector.add(
            "gpt_bad_response_type",
            item_index,
            message_index,
            detail=type(response).__name__,
            message=message,
        )
    else:
        response_clean = response.replace("<sep>", "").strip()
        if len(response_clean) < min_response_chars:
            collector.add(
                "gpt_empty_response",
                item_index,
                message_index,
                detail=f"response_chars={len(response_clean)}",
                message=message,
            )
        if SYSTEM_MARKER_ANY_RE.search(response):
            collector.add("gpt_response_contains_system_marker", item_index, message_index, message=message)
        if "<action>" in response or "</action>" in response:
            collector.add("gpt_response_contains_action_tag", item_index, message_index, message=message)
        if THINK_TAG_RE.search(response):
            collector.add("gpt_response_contains_think_tag", item_index, message_index, message=message)
        if any(fragment in response for fragment in ["slot_values", '"from"', "'from'", '"value"', "BEGIN_META", "BEGIN_FINAL"]):
            collector.add("gpt_response_contains_json_fragment", item_index, message_index, message=message)
        user_norm = normalize_visible_text(current_user_text)
        response_norm = normalize_visible_text(response_clean)
        if user_norm and response_norm and user_norm == response_norm:
            collector.add(
                "gpt_response_copies_current_user",
                item_index,
                message_index,
                detail="response equals visible current human message",
                message=message,
            )

    slot_values = value.get("slot_values")
    if not isinstance(slot_values, dict):
        collector.add(
            "gpt_slot_values_not_dict",
            item_index,
            message_index,
            detail=type(slot_values).__name__,
            message=message,
        )
        return
    for slot_key in REQUIRED_SLOT_KEYS:
        if slot_key not in slot_values:
            collector.add("gpt_missing_slot_key", item_index, message_index, slot_key, message)
        elif not isinstance(slot_values.get(slot_key), str):
            collector.add(
                "gpt_bad_slot_value_type",
                item_index,
                message_index,
                detail=f"{slot_key}={type(slot_values.get(slot_key)).__name__}",
                message=message,
            )


def check_item(
    item: Any,
    item_index: int,
    collector: IssueCollector,
    min_response_chars: int,
) -> None:
    if not isinstance(item, dict):
        collector.add("item_not_dict", item_index, detail=type(item).__name__, message=item)
        return
    if "conversations" not in item:
        collector.add("conversations_missing", item_index)
        return
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        collector.add(
            "conversations_not_list",
            item_index,
            detail=type(conversations).__name__,
            message=conversations,
        )
        return
    if len(conversations) % 2:
        collector.add("odd_message_count", item_index, detail=f"len={len(conversations)}")

    expected_round = 1
    prev_role: str | None = None
    last_human_visible = ""
    for message_index, message in enumerate(conversations):
        if not isinstance(message, dict):
            collector.add(
                "message_not_dict",
                item_index,
                message_index,
                detail=type(message).__name__,
                message=message,
            )
            continue

        role = message.get("from")
        value = message.get("value")
        expected_role = "human" if message_index % 2 == 0 else "gpt"
        if not isinstance(role, str) or role not in {"human", "gpt"}:
            collector.add(
                "bad_from_field",
                item_index,
                message_index,
                detail=repr(role),
                message=message,
            )
        elif role != expected_role:
            collector.add(
                "bad_alternation",
                item_index,
                message_index,
                detail=f"expected={expected_role}, got={role}",
                message=message,
            )
        if isinstance(role, str) and role in {"human", "gpt"} and role == prev_role:
            collector.add("same_role_consecutive", item_index, message_index, f"prev={prev_role}", message)
        if isinstance(role, str) and role in {"human", "gpt"}:
            prev_role = role

        if role == "human":
            if not isinstance(value, str):
                collector.add(
                    "bad_value_type",
                    item_index,
                    message_index,
                    detail=type(value).__name__,
                    message=message,
                )
                continue
            if not value.strip():
                collector.add("empty_message_value", item_index, message_index, message=message)
                continue
            markers = SYSTEM_MARKER_ANY_RE.findall(value)
            strict_markers = SYSTEM_MARKER_STRICT_RE.findall(value)
            if not markers:
                collector.add("human_missing_system_marker", item_index, message_index, message=message)
            else:
                if not strict_markers:
                    collector.add(
                        "human_nonstandard_system_marker",
                        item_index,
                        message_index,
                        detail=",".join(markers),
                        message=message,
                    )
                if len(markers) > 1:
                    collector.add(
                        "human_multiple_system_markers",
                        item_index,
                        message_index,
                        detail=",".join(markers),
                        message=message,
                    )
                round_num = int(markers[-1])
                if round_num != expected_round:
                    collector.add(
                        "human_round_not_sequential",
                        item_index,
                        message_index,
                        detail=f"expected={expected_round}, got={round_num}",
                        message=message,
                    )
                expected_round = round_num + 1
            last_human_visible = visible_human_text(value)
            if not last_human_visible:
                collector.add(
                    "human_empty_visible_content",
                    item_index,
                    message_index,
                    detail="only marker/action/sep",
                    message=message,
                )
        elif role == "gpt":
            check_gpt_value(
                value,
                item_index,
                message_index,
                last_human_visible,
                collector,
                min_response_chars,
                message,
            )


def write_outputs(
    output_prefix: str,
    input_path: str,
    total_items: int,
    collector: IssueCollector,
) -> None:
    issue_jsonl = f"{output_prefix}.object_quality_issues.jsonl"
    report_path = f"{output_prefix}.object_quality_report.md"
    severe_index_path = f"{output_prefix}.object_severe_bad_item_indices.txt"
    any_issue_items = {record["item_index"] for record in collector.records}
    severe_items = {
        record["item_index"]
        for record in collector.records
        if record["issue"] in SEVERE_ISSUES
    }

    with open(issue_jsonl, "w", encoding="utf-8") as f:
        for record in collector.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(severe_index_path, "w", encoding="utf-8") as f:
        for item_index in sorted(severe_items):
            f.write(f"{item_index}\n")

    lines: list[str] = []
    lines.append("# Object-style hard rewrite quality report")
    lines.append("")
    lines.append(f"- source: `{input_path}`")
    lines.append(f"- total_items: {total_items}")
    lines.append(f"- issue_records: {len(collector.records)}")
    lines.append(f"- items_with_any_issue: {len(any_issue_items)}")
    lines.append(f"- severe_bad_items: {len(severe_items)}")
    lines.append(f"- clean_items_by_all_rules: {total_items - len(any_issue_items)}")
    lines.append(f"- clean_items_if_drop_severe_only: {total_items - len(severe_items)}")
    lines.append(f"- issue_jsonl: `{issue_jsonl}`")
    lines.append(f"- severe_bad_item_indices: `{severe_index_path}`")
    lines.append("")
    lines.append("## Issue counts")
    for issue, count in collector.counts.most_common():
        lines.append(f"- {issue}: {count} records, {len(collector.items[issue])} items")
    lines.append("")
    lines.append("## Sample previews")
    for issue, _ in collector.counts.most_common():
        lines.append(f"### {issue}")
        for sample in collector.samples[issue]:
            lines.append("```json")
            lines.append(json.dumps(sample, ensure_ascii=False, indent=2))
            lines.append("```")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {report_path}")
    print(f"wrote {issue_jsonl}")
    print(f"wrote {severe_index_path}")
    print(f"total_items {total_items}")
    print(f"issue_records {len(collector.records)}")
    print(f"items_with_any_issue {len(any_issue_items)}")
    print(f"severe_bad_items {len(severe_items)}")
    for issue, count in collector.counts.most_common():
        print(f"{issue} {count} items {len(collector.items[issue])}")


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_prefix = args.output_prefix or os.path.splitext(input_path)[0]
    items = load_items(input_path)
    collector = IssueCollector(sample_limit=args.sample_limit)
    for item_index, item in enumerate(items):
        check_item(item, item_index, collector, args.min_response_chars)
    write_outputs(output_prefix, input_path, len(items), collector)


if __name__ == "__main__":
    main()
