#!/usr/bin/env python3
"""
Build comparison JSON with fields: question, outputs, gts.

It matches questions from:
- train parquet: extra_info.question + ground_truth
- genrm jsonl: question + output

Output format (grouped by question):
[
  {
    "question": "...",
    "outputs": ["...", "..."],
    "gts": "..."
  }
]
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def normalize_question(q: str) -> str:
    return (q or "").strip()


def load_question_to_gts(parquet_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    table = pq.read_table(str(parquet_path), columns=["ground_truth", "extra_info"])
    gts_list = table.column("ground_truth").to_pylist()
    extra_list = table.column("extra_info").to_pylist()

    q2gts: dict[str, str] = {}
    stats = {
        "rows": 0,
        "valid_question_rows": 0,
        "duplicate_question": 0,
        "duplicate_conflict": 0,
    }

    for gts, extra in zip(gts_list, extra_list):
        stats["rows"] += 1
        if not isinstance(extra, dict):
            continue
        q = normalize_question(str(extra.get("question", "")))
        if not q:
            continue
        stats["valid_question_rows"] += 1

        gts_text = str(gts or "")
        if q in q2gts:
            stats["duplicate_question"] += 1
            if q2gts[q] != gts_text:
                stats["duplicate_conflict"] += 1
            continue

        q2gts[q] = gts_text

    return q2gts, stats


def build_compare_records(genrm_jsonl: Path, q2gts: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    stats = {
        "genrm_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "questions_out": 0,
    }

    with genrm_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["genrm_rows"] += 1

            try:
                obj = json.loads(line)
            except Exception:
                stats["unmatched_rows"] += 1
                continue

            q = normalize_question(str(obj.get("question", "")))
            output = str(obj.get("output", ""))
            if not q or not output:
                stats["unmatched_rows"] += 1
                continue

            gts = q2gts.get(q)
            if gts is None:
                stats["unmatched_rows"] += 1
                continue

            stats["matched_rows"] += 1

            if q not in grouped:
                grouped[q] = {
                    "question": q,
                    "outputs": [],
                    "gts": gts,
                }

            grouped[q]["outputs"].append(output)

    records = list(grouped.values())
    stats["questions_out"] = len(records)
    return records, stats


def dedup_outputs(records: list[dict[str, Any]]) -> None:
    for rec in records:
        seen = set()
        uniq = []
        for out in rec.get("outputs", []):
            if out in seen:
                continue
            seen.add(out)
            uniq.append(out)
        rec["outputs"] = uniq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-parquet",
        default="/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k_mid_stage/single_turn_rl_random_rounds_mid_stage.train.parquet",
    )
    parser.add_argument(
        "--genrm-jsonl",
        default="/data1/chengch/verl_outputs/grpo_single_turn/qwen3_8b_4gpu_stage5_mid_reward_${RUN_TAG}/genrm_io.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="/data1/chengch/verl_outputs/grpo_single_turn/qwen3_8b_4gpu_stage5_mid_reward_${RUN_TAG}/question_outputs_gts.json",
    )
    parser.add_argument(
        "--keep-duplicate-outputs",
        action="store_true",
        help="Keep repeated outputs per question; default removes duplicates while preserving order.",
    )
    args = parser.parse_args()

    train_parquet = Path(args.train_parquet)
    genrm_jsonl = Path(args.genrm_jsonl)
    output_json = Path(args.output_json)

    if not train_parquet.exists():
        raise FileNotFoundError(f"train parquet not found: {train_parquet}")
    if not genrm_jsonl.exists():
        raise FileNotFoundError(f"genrm jsonl not found: {genrm_jsonl}")

    q2gts, parquet_stats = load_question_to_gts(train_parquet)
    records, match_stats = build_compare_records(genrm_jsonl, q2gts)

    if not args.keep_duplicate_outputs:
        dedup_outputs(records)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("[done] wrote:", output_json)
    print("[parquet_stats]", parquet_stats)
    print("[match_stats]", match_stats)


if __name__ == "__main__":
    main()
