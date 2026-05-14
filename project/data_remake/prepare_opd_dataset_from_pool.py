#!/usr/bin/env python3
"""Prepare downloaded prompt pool into OPD-ready dataset + sampling docs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrepareRule:
    min_text_chars: int = 8
    max_text_chars: int = 12_000
    max_messages: int = 20
    min_user_messages: int = 1


RELAXED_RULE = PrepareRule(min_text_chars=1, max_text_chars=20_000, max_messages=40, min_user_messages=1)


REASONING_SOURCES = {"openr1_math", "numina_tir", "gsm8k"}


def _normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt", "bot"}:
        return "assistant"
    return role


def _clean_messages(messages: list[dict[str, Any]], rule: PrepareRule) -> list[dict[str, str]] | None:
    if not isinstance(messages, list) or len(messages) == 0:
        return None
    if len(messages) > rule.max_messages:
        return None

    cleaned: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        role = _normalize_role(role)
        content = content.strip()
        if not content:
            continue
        if len(content) < rule.min_text_chars or len(content) > rule.max_text_chars:
            continue
        cleaned.append({"role": role, "content": content})

    while cleaned and cleaned[-1]["role"] == "assistant":
        cleaned.pop()

    if len(cleaned) == 0:
        return None

    user_cnt = sum(1 for x in cleaned if x["role"] == "user")
    if user_cnt < rule.min_user_messages:
        return None

    return cleaned


def _prompt_to_messages(prompt: Any, prompt_type: str, rule: PrepareRule) -> list[dict[str, str]] | None:
    if prompt_type == "messages":
        if not isinstance(prompt, list):
            return None
        return _clean_messages(prompt, rule)

    if prompt_type == "text":
        if not isinstance(prompt, str):
            return None
        prompt = prompt.strip()
        if len(prompt) < rule.min_text_chars or len(prompt) > rule.max_text_chars:
            return None
        return [{"role": "user", "content": prompt}]

    if isinstance(prompt, list):
        return _clean_messages(prompt, rule)
    if isinstance(prompt, str):
        prompt = prompt.strip()
        if len(prompt) < rule.min_text_chars or len(prompt) > rule.max_text_chars:
            return None
        return [{"role": "user", "content": prompt}]

    return None


def _make_dedup_key(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OPD prompt pool into OPD-ready parquet.")
    parser.add_argument("--input", required=True, help="Downloaded raw jsonl path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (jsonl/parquet/report will be written here)",
    )
    parser.add_argument("--target-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rule = PrepareRule()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_jsonl = out_dir / "opd_prompt_pool_50k_clean.jsonl"
    opd_jsonl = out_dir / "opd_prompt_pool_50k_opd.jsonl"
    opd_parquet = out_dir / "opd_prompt_pool_50k_opd.parquet"
    report_json = out_dir / "opd_sampling_report.json"
    report_md = out_dir / "opd_sampling_report.md"

    import random

    random.seed(args.seed)

    dropped_reason = Counter()
    source_counter = Counter()
    source_prompt_type_counter = defaultdict(Counter)
    global_dedup = set()

    raw_rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped_reason["invalid_json"] += 1
                continue
            if not isinstance(row, dict):
                dropped_reason["non_dict"] += 1
                continue
            raw_rows.append(row)

    cleaned_rows: list[dict[str, Any]] = []

    def _consume_rows(rows: list[dict[str, Any]], active_rule: PrepareRule, stage_name: str) -> None:
        for row in rows:
            if len(cleaned_rows) >= args.target_size:
                break

            source = str(row.get("data_source", "unknown"))
            prompt_type = str(row.get("prompt_type", "unknown"))
            messages = _prompt_to_messages(row.get("prompt"), prompt_type, active_rule)
            if messages is None:
                dropped_reason[f"invalid_prompt_{stage_name}"] += 1
                continue

            dedup_key = _make_dedup_key(messages)
            if dedup_key in global_dedup:
                dropped_reason[f"dedup_{stage_name}"] += 1
                continue
            global_dedup.add(dedup_key)

            ability = "reasoning" if source in REASONING_SOURCES else "general"
            cleaned_row = {
                "id": row.get("id") or f"{source}_{len(cleaned_rows) + 1:06d}",
                "data_source": source,
                "dataset": row.get("dataset"),
                "config": row.get("config"),
                "split": row.get("split"),
                "ability": ability,
                "prompt": messages,
                "prompt_type": "messages",
            }
            cleaned_rows.append(cleaned_row)
            source_counter[source] += 1
            source_prompt_type_counter[source][prompt_type] += 1

    _consume_rows(raw_rows, rule, "strict")
    strict_rows = len(cleaned_rows)

    relaxed_filled = 0
    if len(cleaned_rows) < args.target_size:
        _consume_rows(raw_rows, RELAXED_RULE, "relaxed")
        relaxed_filled = max(len(cleaned_rows) - strict_rows, 0)

    if len(cleaned_rows) < args.target_size:
        raise RuntimeError(
            f"Prepared rows {len(cleaned_rows)} < target_size {args.target_size} after relaxed refill. "
            "Please increase upstream sampling size."
        )

    if len(cleaned_rows) > args.target_size:
        cleaned_rows = cleaned_rows[: args.target_size]

    with cleaned_jsonl.open("w", encoding="utf-8") as writer:
        for row in cleaned_rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    opd_rows: list[dict[str, Any]] = []
    for row in cleaned_rows:
        opd_rows.append(
            {
                "data_source": row["data_source"],
                "prompt": row["prompt"],
                "ability": row["ability"],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "id": row["id"],
                    "dataset": row.get("dataset"),
                    "config": row.get("config"),
                    "split": row.get("split"),
                },
            }
        )

    with opd_jsonl.open("w", encoding="utf-8") as writer:
        for row in opd_rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(opd_rows)
    pq.write_table(table, opd_parquet)

    ability_counter = Counter(row["ability"] for row in cleaned_rows)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input": str(input_path),
        "target_size": args.target_size,
        "output_clean_jsonl": str(cleaned_jsonl),
        "output_opd_jsonl": str(opd_jsonl),
        "output_opd_parquet": str(opd_parquet),
        "final_rows": len(cleaned_rows),
        "strict_rows": strict_rows,
        "relaxed_refill_rows": relaxed_filled,
        "ability_counter": dict(ability_counter),
        "source_counter": dict(source_counter),
        "dropped_reason": dict(dropped_reason),
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    md_lines = [
        "# OPD 50k Prompt Sampling Report",
        "",
        "## Summary",
        f"- Input: `{input_path}`",
        f"- Final rows: **{len(cleaned_rows)}**",
        f"- OPD JSONL: `{opd_jsonl}`",
        f"- OPD Parquet: `{opd_parquet}`",
        "",
        "## Ability Mix",
    ]

    total = max(len(cleaned_rows), 1)
    for key in ["general", "reasoning"]:
        cnt = ability_counter.get(key, 0)
        md_lines.append(f"- {key}: {cnt} ({cnt / total * 100:.2f}%)")

    md_lines += ["", "## Source Breakdown"]
    for source, cnt in sorted(source_counter.items(), key=lambda x: (-x[1], x[0])):
        md_lines.append(f"- {source}: {cnt}")

    md_lines += ["", "## Dropped Rows"]
    if dropped_reason:
        for reason, cnt in sorted(dropped_reason.items(), key=lambda x: (-x[1], x[0])):
            md_lines.append(f"- {reason}: {cnt}")
    else:
        md_lines.append("- none")

    md_lines += [
        "",
        "## Notes",
        "- This dataset is prompt-only and intended for on-policy distillation rollout.",
        "- `reward_model.ground_truth` is intentionally empty placeholder for OPD usage.",
        "- Prompt schema is converted to `prompt: [{role, content}, ...]` for direct verl RL dataset loading.",
    ]

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] final_rows={len(cleaned_rows)}")
    print(f"- clean jsonl: {cleaned_jsonl}")
    print(f"- opd jsonl:   {opd_jsonl}")
    print(f"- opd parquet: {opd_parquet}")
    print(f"- report json: {report_json}")
    print(f"- report md:   {report_md}")


if __name__ == "__main__":
    main()
