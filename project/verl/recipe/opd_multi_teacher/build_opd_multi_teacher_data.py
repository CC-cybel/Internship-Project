#!/usr/bin/env python3
"""Build a small mixed OPD dataset with a per-sample teacher route."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_CONTACT_AGE_INPUT = (
    "/data/chengch/project/rl_remake/outputs/single_turn_rl_contact_age_directed_1k_v3/"
    "single_turn_rl_contact_age_directed.all.jsonl"
)
DEFAULT_CONTACT_STAGE_INPUT = (
    "/data/chengch/project/rl_remake/outputs/single_turn_rl_contact_stage_20k/"
    "single_turn_rl_contact_stage.all.jsonl"
)
DEFAULT_MID_INPUT = (
    "/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k_mid_stage/"
    "single_turn_rl_random_rounds_mid_stage.all.jsonl"
)
DEFAULT_OUTPUT_DIR = "/data/chengch/project/verl/recipe/opd_multi_teacher/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-input", default=None, help="Legacy single contact input; overrides the split contact inputs.")
    parser.add_argument("--contact-age-input", default=DEFAULT_CONTACT_AGE_INPUT)
    parser.add_argument("--contact-stage-input", default=DEFAULT_CONTACT_STAGE_INPUT)
    parser.add_argument("--mid-input", default=DEFAULT_MID_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contact-size", type=int, default=2000, help="Used only with --contact-input.")
    parser.add_argument("--contact-age-size", type=int, default=1000)
    parser.add_argument("--contact-stage-size", type=int, default=1000)
    parser.add_argument("--mid-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def sample_and_tag(rows: list[dict[str, Any]], size: int, route: str, rng: random.Random) -> list[dict[str, Any]]:
    if size > len(rows):
        raise ValueError(f"Requested {size} samples for route {route!r}, but input only has {len(rows)} rows.")
    selected = rng.sample(rows, size)
    tagged: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        item = dict(row)
        extra_info = dict(item.get("extra_info") or {})
        extra_info["teacher_route"] = route
        extra_info["opd_sample_index"] = idx
        item["extra_info"] = extra_info
        item["teacher_route"] = route
        tagged.append(item)
    return tagged


def take_and_tag(rows: list[dict[str, Any]], size: int, route: str, offset: int = 0) -> list[dict[str, Any]]:
    if size > len(rows):
        raise ValueError(f"Requested {size} samples for route {route!r}, but input only has {len(rows)} rows.")
    tagged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:size], start=offset):
        item = dict(row)
        extra_info = dict(item.get("extra_info") or {})
        extra_info["teacher_route"] = route
        extra_info["opd_sample_index"] = idx
        item["extra_info"] = extra_info
        item["teacher_route"] = route
        tagged.append(item)
    return tagged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_path = output_dir / "opd_multi_teacher_10k.train.jsonl"
    val_path = output_dir / "opd_multi_teacher_10k.val.jsonl"
    manifest_path = output_dir / "manifest.json"

    if not args.overwrite and train_path.exists() and val_path.exists():
        print(f"[SKIP] Existing data found: {train_path}")
        print(f"[SKIP] Use --overwrite to rebuild.")
        return

    rng = random.Random(args.seed)
    if args.contact_input:
        contact_rows = read_jsonl(Path(args.contact_input))
        contact = sample_and_tag(contact_rows, args.contact_size, "contact", rng)
        contact_inputs: Any = {"contact_input": args.contact_input}
        contact_sizes: Any = {"contact_size": args.contact_size}
    else:
        contact_age_rows = read_jsonl(Path(args.contact_age_input))
        contact_stage_rows = read_jsonl(Path(args.contact_stage_input))
        contact_age = take_and_tag(contact_age_rows, args.contact_age_size, "contact")
        contact_stage = sample_and_tag(contact_stage_rows, args.contact_stage_size, "contact", rng)
        for idx, item in enumerate(contact_stage, start=len(contact_age)):
            item["extra_info"]["opd_sample_index"] = idx
        contact = contact_age + contact_stage
        contact_inputs = {
            "contact_input": [args.contact_age_input, args.contact_stage_input],
            "contact_age_input": args.contact_age_input,
            "contact_stage_input": args.contact_stage_input,
        }
        contact_sizes = {
            "contact_age_size": args.contact_age_size,
            "contact_stage_size": args.contact_stage_size,
            "contact_size": len(contact),
        }

    mid_rows = read_jsonl(Path(args.mid_input))
    mid = sample_and_tag(mid_rows, args.mid_size, "mid", rng)
    rows = contact + mid
    rng.shuffle(rows)

    val_size = min(args.val_size, len(rows))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        **contact_inputs,
        "mid_input": args.mid_input,
        **contact_sizes,
        "mid_size": args.mid_size,
        "val_size": val_size,
        "seed": args.seed,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "teacher_routes": {
            "contact": "/data1/chengch/models/qwen3_8b_merged",
            "mid": "/data1/chengch/models/qwen3_8b_mid_short_step500",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] train={train_path} rows={len(train_rows)}")
    print(f"[OK] val={val_path} rows={len(val_rows)}")
    print(f"[OK] manifest={manifest_path}")


if __name__ == "__main__":
    main()
