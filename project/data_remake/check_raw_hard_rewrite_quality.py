#!/usr/bin/env python3
"""Quality checks for raw hard-rewrite dialogue JSON files.

The raw files are expected to contain items with:
  - system or system_prompt: prompt text
  - conversations: alternating human/gpt messages

The script writes:
  - *.quality_report.md
  - *.quality_issues.jsonl
  - *.severe_bad_item_indices.txt
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from typing import Any


DEFAULT_INPUT = (
    "/data/chengch/project/data_remake/outputs/raw_hard_rewrite_v2/"
    "hard_reverse_tongyi_v2_action.json"
)

SYSTEM_MARKER_ANY_RE = re.compile(r"【系统(?:数据|状态)：当前\s*第?\s*(\d+)\s*轮】")
SYSTEM_MARKER_STRICT_RE = re.compile(r"【系统数据：当前第\s*(\d+)\s*轮】")
ACTION_RE = re.compile(r"<action>.*?</action>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

SYSTEM_PROMPT_SUSPICIOUS_PATTERNS = {
    "system_contains_conversation_key": re.compile(r'"?conversations"?\s*[:=]'),
    "system_contains_role_field": re.compile(r'"?from"?\s*[:=]\s*"?(?:human|gpt)"?'),
    "system_contains_value_field": re.compile(r'"?value"?\s*[:=]'),
    "system_contains_system_round_marker": SYSTEM_MARKER_ANY_RE,
    "system_contains_begin_final": re.compile(r"BEGIN_FINAL|END_FINAL|BEGIN_META|END_META"),
}


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
    "gpt_empty_visible_reply",
    "gpt_contains_system_round_marker",
    "gpt_contains_action_tag",
    "human_empty_visible_content",
    "system_prompt_missing",
    "system_prompt_not_string",
    "system_prompt_empty",
    "system_prompt_too_short",
    "system_contains_conversation_key",
    "system_contains_role_field",
    "system_contains_value_field",
    "system_contains_system_round_marker",
    "system_contains_begin_final",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check raw hard rewrite JSON data quality."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON file.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to input path without .json.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of previews per issue in the markdown report.",
    )
    parser.add_argument(
        "--min-system-chars",
        type=int,
        default=500,
        help="Flag system prompts shorter than this as suspicious.",
    )
    parser.add_argument(
        "--strict-start-human",
        action="store_true",
        default=True,
        help="Require conversations to start with human and then alternate.",
    )
    return parser.parse_args()


def short_preview(value: Any, limit: int = 500) -> str:
    if isinstance(value, str):
        return value[:limit]
    return repr(value)[:limit]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_items(path: str) -> list[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Input must be a JSON list or a JSON object with list field 'items'.")


class IssueCollector:
    def __init__(self, sample_limit: int) -> None:
        self.issue_counts: collections.Counter[str] = collections.Counter()
        self.issue_items: dict[str, set[int]] = collections.defaultdict(set)
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
        self.issue_counts[issue] += 1
        self.issue_items[issue].add(item_index)
        record: dict[str, Any] = {"issue": issue, "item_index": item_index}
        if message_index is not None:
            record["message_index"] = message_index
        if detail:
            record["detail"] = detail
        if isinstance(message, dict):
            record["from"] = message.get("from")
            record["value_preview"] = short_preview(message.get("value"))
        elif message is not None:
            record["value_preview"] = short_preview(message)
        self.records.append(record)
        if len(self.samples[issue]) < self.sample_limit:
            self.samples[issue].append(record)


def get_system_prompt(item: dict[str, Any]) -> tuple[str | None, Any]:
    if "system_prompt" in item:
        return "system_prompt", item.get("system_prompt")
    if "system" in item:
        return "system", item.get("system")
    return None, None


def visible_human_text(value: str) -> str:
    text = SYSTEM_MARKER_ANY_RE.sub("", value)
    text = ACTION_RE.sub("", text)
    return text.replace("<sep>", "").strip()


def visible_gpt_text(value: str) -> str:
    text = THINK_RE.sub("", value)
    return text.replace("<sep>", "").strip()


def check_system_prompt(
    item: dict[str, Any],
    item_index: int,
    collector: IssueCollector,
    min_system_chars: int,
    system_hashes: collections.Counter[str],
    system_lengths: list[int],
) -> None:
    key, system_prompt = get_system_prompt(item)
    if key is None:
        collector.add("system_prompt_missing", item_index)
        return
    if not isinstance(system_prompt, str):
        collector.add(
            "system_prompt_not_string",
            item_index,
            detail=f"{key}={type(system_prompt).__name__}",
            message=system_prompt,
        )
        return
    stripped = system_prompt.strip()
    system_hashes[stable_hash(stripped)] += 1
    system_lengths.append(len(stripped))
    if not stripped:
        collector.add("system_prompt_empty", item_index, detail=key)
        return
    if len(stripped) < min_system_chars:
        collector.add(
            "system_prompt_too_short",
            item_index,
            detail=f"{key}_chars={len(stripped)}",
            message=stripped,
        )
    for issue, pattern in SYSTEM_PROMPT_SUSPICIOUS_PATTERNS.items():
        match = pattern.search(stripped)
        if match:
            collector.add(
                issue,
                item_index,
                detail=f"{key}; match={match.group(0)[:80]}",
                message=stripped,
            )


def check_conversations(
    item: dict[str, Any],
    item_index: int,
    collector: IssueCollector,
    strict_start_human: bool,
) -> None:
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
    prev_from: str | None = None
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
        if role not in {"human", "gpt"}:
            collector.add(
                "bad_from_field",
                item_index,
                message_index,
                detail=repr(role),
                message=message,
            )
        elif strict_start_human and role != expected_role:
            collector.add(
                "bad_alternation",
                item_index,
                message_index,
                detail=f"expected={expected_role}, got={role}",
                message=message,
            )
        if role in {"human", "gpt"} and prev_from == role:
            collector.add(
                "same_role_consecutive",
                item_index,
                message_index,
                detail=f"prev={prev_from}",
                message=message,
            )
        if role in {"human", "gpt"}:
            prev_from = role

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
        if role == "human":
            if not markers:
                collector.add(
                    "human_missing_system_round_marker",
                    item_index,
                    message_index,
                    message=message,
                )
            else:
                if not strict_markers:
                    collector.add(
                        "human_nonstandard_system_round_marker",
                        item_index,
                        message_index,
                        detail=",".join(markers),
                        message=message,
                    )
                if len(markers) > 1:
                    collector.add(
                        "human_multiple_system_round_markers",
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
            if not visible_human_text(value):
                collector.add(
                    "human_empty_visible_content",
                    item_index,
                    message_index,
                    detail="only marker/action/sep",
                    message=message,
                )
        elif role == "gpt":
            if markers:
                collector.add(
                    "gpt_contains_system_round_marker",
                    item_index,
                    message_index,
                    detail=",".join(markers),
                    message=message,
                )
            if "<action>" in value or "</action>" in value:
                collector.add(
                    "gpt_contains_action_tag",
                    item_index,
                    message_index,
                    message=message,
                )
            if value.count("<think>") != value.count("</think>"):
                collector.add(
                    "gpt_malformed_think_tag",
                    item_index,
                    message_index,
                    detail=f"open={value.count('<think>')}, close={value.count('</think>')}",
                    message=message,
                )
            if value.startswith("<think>") and "</think>" not in value:
                collector.add(
                    "gpt_malformed_think_tag",
                    item_index,
                    message_index,
                    detail="starts with <think> but no closing tag",
                    message=message,
                )
            if not visible_gpt_text(value):
                collector.add(
                    "gpt_empty_visible_reply",
                    item_index,
                    message_index,
                    detail="empty after removing think/sep",
                    message=message,
                )


def write_outputs(
    output_prefix: str,
    input_path: str,
    total_items: int,
    collector: IssueCollector,
    system_hashes: collections.Counter[str],
    system_lengths: list[int],
) -> None:
    issue_jsonl = f"{output_prefix}.quality_issues.jsonl"
    report_path = f"{output_prefix}.quality_report.md"
    severe_index_path = f"{output_prefix}.severe_bad_item_indices.txt"

    severe_items = {
        record["item_index"]
        for record in collector.records
        if record["issue"] in SEVERE_ISSUES
    }
    any_issue_items = {record["item_index"] for record in collector.records}

    with open(issue_jsonl, "w", encoding="utf-8") as f:
        for record in collector.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(severe_index_path, "w", encoding="utf-8") as f:
        for item_index in sorted(severe_items):
            f.write(f"{item_index}\n")

    lines: list[str] = []
    lines.append("# Raw hard rewrite quality report")
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
    lines.append("## System prompt summary")
    lines.append(f"- unique_system_prompt_hashes: {len(system_hashes)}")
    if system_lengths:
        lengths = sorted(system_lengths)
        lines.append(f"- min_system_chars: {lengths[0]}")
        lines.append(f"- p50_system_chars: {lengths[len(lengths) // 2]}")
        lines.append(f"- max_system_chars: {lengths[-1]}")
    lines.append("- top_system_prompt_hashes:")
    for hash_value, count in system_hashes.most_common(10):
        lines.append(f"  - {hash_value}: {count}")
    lines.append("")
    lines.append("## Issue counts")
    for issue, count in collector.issue_counts.most_common():
        lines.append(
            f"- {issue}: {count} records, "
            f"{len(collector.issue_items[issue])} items"
        )
    lines.append("")
    lines.append("## Sample previews")
    for issue, _ in collector.issue_counts.most_common():
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
    for issue, count in collector.issue_counts.most_common():
        print(f"{issue} {count} items {len(collector.issue_items[issue])}")


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(input_path)[0]

    items = load_items(input_path)
    collector = IssueCollector(sample_limit=args.sample_limit)
    system_hashes: collections.Counter[str] = collections.Counter()
    system_lengths: list[int] = []

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            collector.add(
                "item_not_dict",
                item_index,
                detail=type(item).__name__,
                message=item,
            )
            continue
        check_system_prompt(
            item,
            item_index,
            collector,
            args.min_system_chars,
            system_hashes,
            system_lengths,
        )
        check_conversations(
            item,
            item_index,
            collector,
            strict_start_human=args.strict_start_human,
        )

    write_outputs(
        output_prefix,
        input_path,
        len(items),
        collector,
        system_hashes,
        system_lengths,
    )


if __name__ == "__main__":
    main()
