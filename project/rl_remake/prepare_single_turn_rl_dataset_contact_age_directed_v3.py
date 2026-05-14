#!/usr/bin/env python3
"""Prepare 1k contact-stage RL data with deterministic age-directed contact rules.

This version follows the contact-stage v2 data shape, but changes the system
prompt with a deterministic text rewrite:
- filter out any sample mentioning "孩子"
- rewrite [留联触发] as [留联触发与分龄策略]
- rewrite the conversion excuse to avoid phone-first wording
- replace the old phone-first downgrade strategy with age-aware fallback wording
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_single_turn_rl_dataset_contact_stage_v2 import (
    SOURCES,
    SOURCE_PATHS,
    build_rows_for_source,
    choose_contact_turn_ids,
    sample_balanced,
    summarize_rows,
    transform_system_prompt as transform_slot_schema_only,
    write_jsonl,
    write_parquet,
)


AGE_CONTACT_SENTENCE = (
    "此时必须执行分龄定向策略：若已知用户年龄大于 35 岁，必须优先索要电话号码；"
    "若用户年龄小于等于 35 岁或年龄未知，必须优先索要微信号。"
)

CONTACT_TRIGGER_LINE_RE = re.compile(
    r"(?P<prefix>(?:[-\d. ]*)?)\[留联触发(?:与分龄策略)?\]：(?P<body>[^\n]*)"
)
CONVERSION_EXCUSE_LINE_RE = re.compile(r"(?:[-\d. ]*)?转化借口：[^\n]+")
NEW_CONVERSION_EXCUSE = (
    "转化借口：根据用户的意图和症状，以“详细讲解成因”、“后期应对方案”"
    "及“一对一免费建议指导”为钩子引导留联。"
)
DOWNGRADE_LINE_RE = re.compile(r"(?:[-\d. ]*)?(?:降级策略|拒绝处理)：[^\n]+")
NEW_DOWNGRADE = (
    "- 降级策略：若用户拒绝提供首选联系方式，以“名额保留”或“医疗风险”为由进行最后挽留，"
    "并尝试切换另一种联系方式（如先要微信被拒，可尝试要电话，反之亦然，但需符合年龄逻辑）。"
)


def rewrite_contact_trigger_line(match: re.Match[str]) -> str:
    body = match.group("body")
    trigger_phrase = "强制启动首次留联尝试"
    trigger_end = body.find(trigger_phrase)
    if trigger_end >= 0:
        trigger_end += len(trigger_phrase)
        kept_body = body[:trigger_end].rstrip("，,。；; ")
    else:
        kept_body = body.rstrip("，,。；; ")

    return f"{match.group('prefix')}[留联触发与分龄策略]：{kept_body}。{AGE_CONTACT_SENTENCE}"


def transform_age_directed_prompt(system_prompt: str) -> str:
    result = transform_slot_schema_only(system_prompt, use_age_preference=False)

    result = CONTACT_TRIGGER_LINE_RE.sub(rewrite_contact_trigger_line, result, count=1)

    result = CONVERSION_EXCUSE_LINE_RE.sub(NEW_CONVERSION_EXCUSE, result)
    result = DOWNGRADE_LINE_RE.sub(NEW_DOWNGRADE, result)
    result = result.replace("号。。", "号。")
    return result


def contains_child_marker(row: dict[str, Any]) -> bool:
    texts: list[str] = []
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("original_system_prompt", "transformed_system_prompt", "question"):
            value = extra.get(key)
            if isinstance(value, str):
                texts.append(value)
        conv = extra.get("conversations")
        if isinstance(conv, list):
            for msg in conv:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    texts.append(msg["content"])

    if isinstance(row.get("ground_truth"), str):
        texts.append(row["ground_truth"])
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                texts.append(msg["content"])

    return any("孩子" in text for text in texts)


def apply_age_directed_transform(row: dict[str, Any]) -> None:
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        return

    original_system_prompt = str(extra.get("original_system_prompt", ""))
    transformed_system_prompt = transform_age_directed_prompt(original_system_prompt)
    extra["transformed_system_prompt"] = transformed_system_prompt
    extra["use_age_preference"] = True
    extra["age_directed_contact_v3"] = True
    extra["filtered_child_marker"] = False
    extra["prompt_rewrite_rule"] = {
        "contact_trigger_label": "[留联触发与分龄策略]",
        "age_contact_sentence": AGE_CONTACT_SENTENCE,
        "conversion_excuse": NEW_CONVERSION_EXCUSE,
        "downgrade_strategy": NEW_DOWNGRADE,
    }

    if row.get("prompt") and isinstance(row["prompt"], list) and row["prompt"][0].get("role") == "system":
        row["prompt"][0]["content"] = transformed_system_prompt

    ground_truth = str(row.get("ground_truth", ""))
    transformed_ground_truth = transform_slot_schema_only(ground_truth, use_age_preference=False)
    row["ground_truth"] = transformed_ground_truth

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        reward_model["ground_truth"] = transformed_ground_truth
        reward_model["style"] = "contact_stage_age_directed_v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-json", default=SOURCE_PATHS["hard"])
    parser.add_argument("--normal-json", default=SOURCE_PATHS["normal"])
    parser.add_argument(
        "--output-dir",
        default="/data/chengch/project/rl_remake/outputs/single_turn_rl_contact_age_directed_1k_v3",
    )
    parser.add_argument("--total-samples", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-contact-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {"hard": args.hard_json, "normal": args.normal_json}
    collected = {
        src: choose_contact_turn_ids(
            source=src,
            path=source_paths[src],
            strict_contact_signal=args.strict_contact_signal,
        )
        for src in SOURCES
    }

    rows_all: list[dict[str, Any]] = []
    for src in SOURCES:
        rows_all.extend(
            build_rows_for_source(
                source=src,
                path=source_paths[src],
                selected_ids=collected[src]["selected_ids"],
                age_preference_indices=set(),
                row_offset=len(rows_all),
            )
        )

    before_child_filter = len(rows_all)
    rows_all = [row for row in rows_all if not contains_child_marker(row)]
    child_filtered = before_child_filter - len(rows_all)
    if not rows_all:
        raise RuntimeError("No rows left after filtering child-marker samples.")

    rows = sample_balanced(rows_all, total=args.total_samples, seed=args.seed)
    for row in rows:
        apply_age_directed_transform(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = min(max(args.val_size, 0), len(rows))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    all_jsonl = output_dir / "single_turn_rl_contact_age_directed.all.jsonl"
    train_jsonl = output_dir / "single_turn_rl_contact_age_directed.train.jsonl"
    val_jsonl = output_dir / "single_turn_rl_contact_age_directed.val.jsonl"
    write_jsonl(all_jsonl, rows)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    all_parquet = output_dir / "single_turn_rl_contact_age_directed.all.parquet"
    train_parquet = output_dir / "single_turn_rl_contact_age_directed.train.parquet"
    val_parquet = output_dir / "single_turn_rl_contact_age_directed.val.parquet"
    parquet_ok = write_parquet(all_parquet, rows)
    parquet_ok = write_parquet(train_parquet, train_rows) and parquet_ok
    parquet_ok = write_parquet(val_parquet, val_rows) and parquet_ok

    stats = {
        "seed": args.seed,
        "targets": {
            "total_samples": args.total_samples,
            "val_size": val_size,
            "strict_contact_signal": args.strict_contact_signal,
        },
        "filters": {
            "child_marker": "孩子",
            "candidate_rows_before_child_filter": before_child_filter,
            "child_filtered": child_filtered,
            "candidate_rows_after_child_filter": len(rows_all),
        },
        "selected_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "by_source": dict(Counter(row["data_source"] for row in rows)),
            "turn_round": dict(Counter(row["extra_info"].get("turn_round") for row in rows)),
            "age_directed_contact_v3": sum(
                1 for row in rows if row.get("extra_info", {}).get("age_directed_contact_v3")
            ),
        },
        "candidate_stats": {src: dict(collected[src]["stats"]) for src in SOURCES},
        "split_stats": {
            "all": summarize_rows(rows),
            "train": summarize_rows(train_rows),
            "val": summarize_rows(val_rows),
        },
        "outputs": {
            "all_jsonl": str(all_jsonl),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "all_parquet": str(all_parquet) if parquet_ok else None,
            "train_parquet": str(train_parquet) if parquet_ok else None,
            "val_parquet": str(val_parquet) if parquet_ok else None,
        },
    }
    stats_path = output_dir / "single_turn_rl_contact_age_directed.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote dataset:")
    print(f"  all:   {all_jsonl}")
    print(f"  train: {train_jsonl}")
    print(f"  val:   {val_jsonl}")
    if parquet_ok:
        print(f"  parquet all/train/val written under {output_dir}")
    else:
        print("  parquet: skipped (package 'datasets' not installed)")
    print(f"  stats: {stats_path}")
    print(
        "[stats] rows all/train/val="
        f"{len(rows)}/{len(train_rows)}/{len(val_rows)}, child_filtered={child_filtered}"
    )


if __name__ == "__main__":
    main()
