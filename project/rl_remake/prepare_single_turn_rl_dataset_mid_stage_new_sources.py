#!/usr/bin/env python3
"""Prepare 10k mid-stage single-turn RL data from the new contact-stage sources.

Data sources are shared with prepare_single_turn_rl_dataset_contact_stage_new_sources.py:
- hard_rewrite_v2_sft_score4_5_clean_dual_full.jsonl
- anti_hijack_abc_array.json

Selection logic follows prepare_single_turn_rl_dataset_mid_stage.py:
- start stage (turn_round <= 2): skip
- mid stage (3 <= turn_round < contact_round): keep
- contact stage (turn_round >= contact_round): skip
- randomly choose one eligible mid-stage assistant turn per conversation

Default output is 10k rows, balanced as 5k hard_sft + 5k anti_hijack.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_PATHS = {
    "hard_sft": "/data/chengch/project/data_remake/runs/hard_sft_stage1/hard_rewrite_v2_sft_score4_5_clean_dual_full.jsonl",
    "anti_hijack": "/data/chengch/project/data_remake/runs/last_turn_value_slots_split/anti_hijack_abc_array.json",
}
SOURCES = tuple(SOURCE_PATHS)

TURN_ROUND_PATTERN = re.compile(r"当前第\s*(\d+)\s*轮")
CONTACT_ROUND_PATTERNS = (
    re.compile(r"留联触发[^\n。]*?第\s*(\d+)\s*轮"),
    re.compile(r"强制启动首次留联[^\n。]*?第\s*(\d+)\s*轮"),
    re.compile(r"首次留联[^\n。]*?第\s*(\d+)\s*轮"),
)
AGE_GENDER_ROUND_PATTERNS = (
    re.compile(r"第\s*(\d+)\s*轮前[^\n。]*?(?:询问|获取)[^\n。]*?年龄[^\n。]*?(?:性别|男女)"),
    re.compile(r"第\s*(\d+)\s*轮前[^\n。]*?(?:询问|获取)[^\n。]*?(?:性别|男女)[^\n。]*?年龄"),
)

FORMAT_PATTERN = re.compile(
    r"^BEGIN_META\n(?P<meta>[\s\S]*?)\nEND_META\nBEGIN_FINAL\n(?P<final>[\s\S]*?)\nEND_FINAL$",
    flags=re.DOTALL,
)


def normalize_role(raw_role: Any) -> str | None:
    role = str(raw_role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def parse_turn_round(text: str) -> int | None:
    match = TURN_ROUND_PATTERN.search(text or "")
    return int(match.group(1)) if match else None


def parse_contact_round(system_prompt: str) -> int | None:
    for pattern in CONTACT_ROUND_PATTERNS:
        match = pattern.search(system_prompt or "")
        if match:
            return int(match.group(1))
    return None


def parse_age_gender_deadline_round(system_prompt: str) -> int | None:
    rounds: list[int] = []
    for pattern in AGE_GENDER_ROUND_PATTERNS:
        rounds.extend(int(match.group(1)) for match in pattern.finditer(system_prompt or ""))
    return min(rounds) if rounds else None


def is_dual_format_valid(text: str) -> bool:
    payload = (text or "").strip()
    if (
        payload.count("BEGIN_META") != 1
        or payload.count("END_META") != 1
        or payload.count("BEGIN_FINAL") != 1
        or payload.count("END_FINAL") != 1
    ):
        return False
    match = FORMAT_PATTERN.match(payload)
    if not match:
        return False
    meta_lines = [line for line in match.group("meta").split("\n") if line.strip()]
    if len(meta_lines) < 2:
        return False
    if not meta_lines[0].startswith("action="):
        return False
    if not meta_lines[1].startswith("thought="):
        return False
    if not all("=" in line for line in meta_lines):
        return False
    if not match.group("final").strip():
        return False
    return True


def load_records(path: str) -> list[dict[str, Any]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array at {path}")
            return data
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected object on {path}:{line_no}")
            records.append(item)
        return records


def collect_mid_stage_turn_ids(
    source: str,
    records: list[dict[str, Any]],
    seed: int,
    default_contact_round: int,
) -> dict[str, Any]:
    print(f"[collect] source={source} conversations={len(records)}")
    by_conv: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    stats = Counter()

    for conv_idx, rec in enumerate(records):
        conv = rec.get("conversations") or []
        if not isinstance(conv, list):
            stats["bad_conversations"] += 1
            continue

        system_prompt = str(rec.get("system", ""))
        contact_round = parse_contact_round(system_prompt)
        if contact_round is None:
            contact_round = default_contact_round
            stats["missing_contact_round_used_default"] += 1

        last_user_round: int | None = None
        for turn_idx, turn in enumerate(conv):
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("from", turn.get("role")))
            content = str(turn.get("value", turn.get("content", "")))
            if role == "user":
                parsed = parse_turn_round(content)
                if parsed is not None:
                    last_user_round = parsed
                continue
            if role != "assistant":
                continue

            stats["assistant_total"] += 1
            if not is_dual_format_valid(content):
                stats["assistant_invalid_format"] += 1
                continue
            stats["format_valid"] += 1

            if last_user_round is None:
                stats["filtered_missing_turn_round"] += 1
            elif last_user_round <= 2:
                stats["filtered_start_stage"] += 1
            elif last_user_round >= contact_round:
                stats["filtered_contact_stage"] += 1
            else:
                by_conv[conv_idx].append((conv_idx, turn_idx, last_user_round))
                stats[f"round_{last_user_round}"] += 1

    rng = random.Random(seed)
    selected_ids: set[tuple[int, int]] = set()
    selected_rounds = Counter()
    for conv_idx, candidates in by_conv.items():
        conv_idx2, turn_idx, turn_round = rng.choice(candidates)
        selected_ids.add((conv_idx2, turn_idx))
        selected_rounds[turn_round] += 1
        stats["picked_conversations"] += 1

    print(
        f"[collect] source={source} selected={len(selected_ids)} "
        f"format_valid={stats['format_valid']} picked_conversations={stats['picked_conversations']}"
    )
    print(
        f"  filtered_start={stats.get('filtered_start_stage', 0)}, "
        f"filtered_contact={stats.get('filtered_contact_stage', 0)}, "
        f"selected_rounds={dict(selected_rounds)}"
    )
    return {"selected_ids": selected_ids, "stats": stats, "selected_rounds": selected_rounds}


def build_rows_for_source(
    source: str,
    records: list[dict[str, Any]],
    selected_ids: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not selected_ids:
        return rows

    for conv_idx, rec in enumerate(records):
        conv = rec.get("conversations") or []
        if not isinstance(conv, list):
            continue
        system_prompt = str(rec.get("system", ""))
        contact_round = parse_contact_round(system_prompt)
        age_deadline = parse_age_gender_deadline_round(system_prompt)

        normalized_history: list[dict[str, str]] = []
        last_user_round: int | None = None
        for turn_idx, turn in enumerate(conv):
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("from", turn.get("role")))
            content = str(turn.get("value", turn.get("content", "")))
            if role is None:
                continue
            if role == "user":
                normalized_history.append({"role": "user", "content": content})
                parsed = parse_turn_round(content)
                if parsed is not None:
                    last_user_round = parsed
                continue
            if role != "assistant":
                continue

            cid = (conv_idx, turn_idx)
            if cid not in selected_ids:
                normalized_history.append({"role": "assistant", "content": content})
                continue
            if not is_dual_format_valid(content):
                normalized_history.append({"role": "assistant", "content": content})
                continue

            history_snapshot = [dict(msg) for msg in normalized_history]
            prompt_messages: list[dict[str, str]] = []
            if system_prompt.strip():
                prompt_messages.append({"role": "system", "content": system_prompt})
            prompt_messages.extend(history_snapshot)

            question = ""
            for hist_turn in reversed(history_snapshot):
                if hist_turn["role"] == "user" and hist_turn["content"].strip():
                    question = hist_turn["content"].strip()
                    break

            sample_id = f"{source}_{conv_idx}_{turn_idx}"
            extra_info = {
                "sample_id": sample_id,
                "source": source,
                "conv_id": conv_idx,
                "turn_id": turn_idx,
                "slice_bucket": "mid_stage",
                "original_system_prompt": system_prompt,
                "rule_contact_round": contact_round,
                "rule_age_gender_deadline_round": age_deadline,
                "turn_round": last_user_round,
                "question": question,
                "conversations": history_snapshot,
                "exclude_last_turn": False,
                "truncation_policy": "mid_stage_random_per_conversation_new_sources",
            }
            reward_model = {
                "style": "mid_stage_rule",
                "ground_truth": content,
                "target_round": last_user_round,
                "contact_round": contact_round,
                "age_gender_deadline_round": age_deadline,
            }
            rows.append(
                {
                    "prompt": prompt_messages,
                    "ground_truth": content,
                    "extra_info": extra_info,
                    "reward_model": reward_model,
                    "data_source": source,
                    "agent_name": "single_turn_agent",
                    "index": sample_id,
                }
            )
            normalized_history.append({"role": "assistant", "content": content})

    print(f"[build] source={source} built={len(rows)}")
    return rows


def sample_per_source(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["data_source"]].append(row)
    for source_rows in by_source.values():
        rng.shuffle(source_rows)

    sampled: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    leftovers: list[dict[str, Any]] = []
    for source, target in targets.items():
        take = min(target, len(by_source[source]))
        sampled.extend(by_source[source][:take])
        if take < target:
            shortages[source] = target - take
        leftovers.extend(by_source[source][take:])

    shortage_total = sum(shortages.values())
    if shortage_total:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:shortage_total])

    rng.shuffle(sampled)
    return sampled[: sum(targets.values())]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from datasets import Dataset  # type: ignore
    except Exception:
        return False
    Dataset.from_list(rows).to_parquet(str(path))
    return True


def _len_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0, "p90": 0, "p99": 0}
    sorted_vals = sorted(values)

    def pct(p: float) -> int:
        idx = int((len(sorted_vals) - 1) * p)
        return sorted_vals[idx]

    return {
        "count": len(sorted_vals),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": round(sum(sorted_vals) / len(sorted_vals), 2),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(row["data_source"] for row in rows)
    by_round = Counter(row["extra_info"].get("turn_round") for row in rows)
    by_contact_round = Counter(row["extra_info"].get("rule_contact_round") for row in rows)
    by_age_deadline = Counter(row["extra_info"].get("rule_age_gender_deadline_round") for row in rows)
    prompt_turn_counts = [len(row.get("prompt", [])) for row in rows]
    prompt_char_lens = [sum(len(msg.get("content", "")) for msg in row.get("prompt", [])) for row in rows]
    answer_char_lens = [len(row.get("ground_truth", "")) for row in rows]
    has_system = sum(
        1
        for row in rows
        if row.get("prompt") and isinstance(row["prompt"], list) and row["prompt"][0].get("role") == "system"
    )
    conv_pairs = {(row["data_source"], row["extra_info"].get("conv_id")) for row in rows}
    turn_round_top10 = sorted(
        ({"turn_round": k, "count": v} for k, v in by_round.items()),
        key=lambda x: (-x["count"], str(x["turn_round"])),
    )[:10]
    return {
        "rows": len(rows),
        "by_source": dict(by_source),
        "unique_conversations": len(conv_pairs),
        "coverage": {
            "system_prompt_ratio": round(has_system / len(rows), 4) if rows else 0.0,
            "contact_round_known_ratio": round(
                sum(1 for row in rows if row["extra_info"].get("rule_contact_round") is not None) / len(rows), 4
            )
            if rows
            else 0.0,
            "age_gender_deadline_known_ratio": round(
                sum(1 for row in rows if row["extra_info"].get("rule_age_gender_deadline_round") is not None)
                / len(rows),
                4,
            )
            if rows
            else 0.0,
        },
        "turn_round": {
            "top10": turn_round_top10,
            "full": {str(k): int(v) for k, v in sorted(by_round.items(), key=lambda kv: str(kv[0]))},
        },
        "rule_contact_round": {str(k): int(v) for k, v in sorted(by_contact_round.items(), key=lambda kv: str(kv[0]))},
        "rule_age_gender_deadline_round": {
            str(k): int(v) for k, v in sorted(by_age_deadline.items(), key=lambda kv: str(kv[0]))
        },
        "length": {
            "prompt_turn_count": _len_stats(prompt_turn_counts),
            "prompt_char_len": _len_stats(prompt_char_lens),
            "answer_char_len": _len_stats(answer_char_lens),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-sft-jsonl", default=SOURCE_PATHS["hard_sft"])
    parser.add_argument("--anti-hijack-json", default=SOURCE_PATHS["anti_hijack"])
    parser.add_argument(
        "--output-dir",
        default="/data/chengch/project/rl_remake/outputs/single_turn_rl_mid_stage_new_sources_10k",
    )
    parser.add_argument("--total-samples", type=int, default=10000)
    parser.add_argument("--hard-sft-samples", type=int, default=5000)
    parser.add_argument("--anti-hijack-samples", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-contact-round", type=int, default=99)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "hard_sft": args.hard_sft_jsonl,
        "anti_hijack": args.anti_hijack_json,
    }
    records_by_source = {source: load_records(path) for source, path in source_paths.items()}
    collected = {
        source: collect_mid_stage_turn_ids(
            source=source,
            records=records_by_source[source],
            seed=args.seed + idx,
            default_contact_round=args.default_contact_round,
        )
        for idx, source in enumerate(SOURCES)
    }

    rows_all: list[dict[str, Any]] = []
    for source in SOURCES:
        rows_all.extend(
            build_rows_for_source(
                source=source,
                records=records_by_source[source],
                selected_ids=collected[source]["selected_ids"],
            )
        )
    if not rows_all:
        raise RuntimeError("No rows built. Check source data and extraction rules.")

    source_targets = {
        "hard_sft": args.hard_sft_samples,
        "anti_hijack": args.anti_hijack_samples,
    }
    if args.total_samples != sum(source_targets.values()):
        per_source = args.total_samples // len(SOURCES)
        source_targets = {source: per_source for source in SOURCES}
        for source in SOURCES[: args.total_samples % len(SOURCES)]:
            source_targets[source] += 1

    rows = sample_per_source(rows_all, targets=source_targets, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = min(max(args.val_size, 0), len(rows))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    prefix = "single_turn_rl_mid_stage_new_sources"
    all_jsonl = output_dir / f"{prefix}.all.jsonl"
    train_jsonl = output_dir / f"{prefix}.train.jsonl"
    val_jsonl = output_dir / f"{prefix}.val.jsonl"
    write_jsonl(all_jsonl, rows)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    all_parquet = output_dir / f"{prefix}.all.parquet"
    train_parquet = output_dir / f"{prefix}.train.parquet"
    val_parquet = output_dir / f"{prefix}.val.parquet"
    parquet_ok = write_parquet(all_parquet, rows)
    parquet_ok = write_parquet(train_parquet, train_rows) and parquet_ok
    parquet_ok = write_parquet(val_parquet, val_rows) and parquet_ok

    stats = {
        "seed": args.seed,
        "targets": {
            "total_samples": args.total_samples,
            "source_targets": source_targets,
            "val_size": val_size,
            "default_contact_round": args.default_contact_round,
        },
        "source_paths": source_paths,
        "candidate_stats": {source: dict(collected[source]["stats"]) for source in SOURCES},
        "selected_candidate_rounds": {source: dict(collected[source]["selected_rounds"]) for source in SOURCES},
        "selected_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "by_source": dict(Counter(row["data_source"] for row in rows)),
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
    stats_path = output_dir / f"{prefix}.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote dataset")
    print(f"  all:   {all_jsonl}")
    print(f"  train: {train_jsonl}")
    print(f"  val:   {val_jsonl}")
    print(f"  stats: {stats_path}")
    if parquet_ok:
        print(f"  parquet all/train/val written under {output_dir}")
    else:
        print("  parquet: skipped (package 'datasets' not installed)")
    print(f"[stats] rows all/train/val={len(rows)}/{len(train_rows)}/{len(val_rows)}")
    print(f"[stats] by_source={dict(Counter(row['data_source'] for row in rows))}")
    print(f"[stats] turn_round={dict(Counter(row['extra_info'].get('turn_round') for row in rows))}")


if __name__ == "__main__":
    main()
