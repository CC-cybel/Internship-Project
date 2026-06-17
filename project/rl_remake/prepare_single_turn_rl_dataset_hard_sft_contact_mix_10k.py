#!/usr/bin/env python3
"""Build a 10k hard-SFT contact-stage RL dataset.

The dataset is sampled from:
  hard_rewrite_v2_sft_score4_5_clean_dual_full.jsonl

Composition:
- 8k plain contact-stage single-turn slices
- 2k age-directed contact-rule slices

Output schema matches verl single-turn RL training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_single_turn_rl_dataset_contact_age_directed_v3 import (
    apply_age_directed_transform,
    contains_child_marker,
)
from prepare_single_turn_rl_dataset_contact_stage_new_sources import (
    build_rows_for_source,
    choose_contact_turn_ids,
    load_records,
    sample_balanced,
    summarize_rows,
    write_jsonl,
    write_parquet,
)


DEFAULT_SOURCE = (
    "/data/chengch/project/data_remake/runs/hard_sft_stage1/"
    "hard_rewrite_v2_sft_score4_5_clean_dual_full.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data/chengch/project/rl_remake/outputs/"
    "single_turn_rl_hard_sft_contact_mix_10k"
)


def relabel_row(row: dict[str, Any], variant: str, serial: int, source_file: str) -> None:
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        original_sample_id = extra.get("sample_id")
        extra["original_sample_id"] = original_sample_id
        extra["slice_bucket"] = f"contact_stage_{variant}"
        extra["dataset_variant"] = variant
        extra["source_file"] = source_file

    row["data_source"] = f"hard_sft_{variant}"
    row["index"] = f"hard_sft_{variant}_{serial:05d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plain-samples", type=int, default=8000)
    parser.add_argument("--age-directed-samples", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-contact-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-contact-round", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.source_jsonl)
    collected = choose_contact_turn_ids(
        source="hard_sft",
        records=records,
        strict_contact_signal=args.strict_contact_signal,
        min_contact_round=args.min_contact_round,
    )
    rows_all = build_rows_for_source(
        source="hard_sft",
        records=records,
        selected_ids=collected["selected_ids"],
        row_offset=0,
    )
    if not rows_all:
        raise RuntimeError("No rows built. Check source data and extraction rules.")

    target_total = args.plain_samples + args.age_directed_samples
    rows_sampled = sample_balanced(rows_all, total=target_total, seed=args.seed)
    if len(rows_sampled) < target_total:
        raise RuntimeError(f"Only sampled {len(rows_sampled)} rows, target is {target_total}.")

    rng = random.Random(args.seed)
    rng.shuffle(rows_sampled)

    plain_rows = rows_sampled[: args.plain_samples]
    age_candidates = rows_sampled[args.plain_samples :]
    before_child_filter = len(age_candidates)
    age_rows = [row for row in age_candidates if not contains_child_marker(row)]
    initial_child_filtered = before_child_filter - len(age_rows)
    replacement_count = 0

    if len(age_rows) < args.age_directed_samples:
        used_ids = {id(row) for row in plain_rows + age_candidates}
        replacements = [
            row
            for row in rows_all
            if id(row) not in used_ids and not contains_child_marker(row)
        ]
        rng.shuffle(replacements)
        replacement_count = args.age_directed_samples - len(age_rows)
        age_rows.extend(replacements[:replacement_count])

    age_rows = age_rows[: args.age_directed_samples]
    if len(age_rows) < args.age_directed_samples:
        raise RuntimeError(
            f"Only found {len(age_rows)} age-directed rows after child filtering, "
            f"target is {args.age_directed_samples}."
        )

    for idx, row in enumerate(plain_rows):
        relabel_row(row, "plain", idx, args.source_jsonl)
    for idx, row in enumerate(age_rows):
        apply_age_directed_transform(row)
        relabel_row(row, "age_directed_v3", idx, args.source_jsonl)

    rows = plain_rows + age_rows
    rng.shuffle(rows)

    val_size = min(max(args.val_size, 0), len(rows))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    all_jsonl = output_dir / "single_turn_rl_hard_sft_contact_mix.all.jsonl"
    train_jsonl = output_dir / "single_turn_rl_hard_sft_contact_mix.train.jsonl"
    val_jsonl = output_dir / "single_turn_rl_hard_sft_contact_mix.val.jsonl"
    write_jsonl(all_jsonl, rows)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    all_parquet = output_dir / "single_turn_rl_hard_sft_contact_mix.all.parquet"
    train_parquet = output_dir / "single_turn_rl_hard_sft_contact_mix.train.parquet"
    val_parquet = output_dir / "single_turn_rl_hard_sft_contact_mix.val.parquet"
    parquet_ok = write_parquet(all_parquet, rows)
    parquet_ok = write_parquet(train_parquet, train_rows) and parquet_ok
    parquet_ok = write_parquet(val_parquet, val_rows) and parquet_ok

    stats = {
        "seed": args.seed,
        "targets": {
            "plain_samples": args.plain_samples,
            "age_directed_samples": args.age_directed_samples,
            "total_samples": target_total,
            "val_size": val_size,
            "strict_contact_signal": args.strict_contact_signal,
            "min_contact_round": args.min_contact_round,
        },
        "source_path": args.source_jsonl,
        "candidate_stats": dict(collected["stats"]),
        "filters": {
            "age_directed_child_marker": "孩子",
            "age_directed_candidates_before_child_filter": before_child_filter,
            "age_directed_child_filtered_initial": initial_child_filtered,
            "age_directed_replacements_after_child_filter": replacement_count,
        },
        "selected_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "by_source": dict(Counter(row["data_source"] for row in rows)),
            "by_variant": dict(
                Counter(row.get("extra_info", {}).get("dataset_variant") for row in rows)
            ),
            "turn_round": dict(Counter(row["extra_info"].get("turn_round") for row in rows)),
        },
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
    stats_path = output_dir / "single_turn_rl_hard_sft_contact_mix.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote dataset")
    print(f"  all:   {all_jsonl}")
    print(f"  train: {train_jsonl}")
    print(f"  val:   {val_jsonl}")
    if parquet_ok:
        print(f"  parquet all/train/val written under {output_dir}")
    else:
        print("  parquet: skipped (package 'datasets' not installed)")
    print(f"  stats: {stats_path}")
    print(f"[stats] rows all/train/val={len(rows)}/{len(train_rows)}/{len(val_rows)}")
    print(f"[stats] by_variant={stats['selected_counts']['by_variant']}")


if __name__ == "__main__":
    main()
