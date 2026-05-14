#!/usr/bin/env python3
"""
Prepare a random-any-round single-turn RL dataset for verl.

Compared with the original pipeline, this script now keeps one random valid turn
from any available round in each conversation.

Output rows keep the RL-compatible schema used by verl GRPO single-turn training.

Output fields:
- prompt: list[{"role": ..., "content": ...}]
- ground_truth: str
- extra_info: dict
- reward_model: dict
- data_source: str
- agent_name: str
- index: str
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset


SOURCE_PATHS = {
    "hard": "/data/wangpf/project/LlamaFactory/data/hard_dual_s5_full.json",
    "normal": "/data/wangpf/project/data_remake/outputs/normal/normal_s5_dual_full_drop_system_keywords.json",
}
SOURCES = ("hard", "normal")

TURN_ROUND_PATTERN = re.compile(r"当前第\s*(\d+)\s*轮")
CONTACT_ROUND_PATTERNS = (
    re.compile(r"留联触发[^\n。]*?第\s*(\d+)\s*轮"),
    re.compile(r"强制启动首次留联[^\n。]*?第\s*(\d+)\s*轮"),
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
    m = TURN_ROUND_PATTERN.search(text or "")
    if m:
        return int(m.group(1))
    return None


def parse_contact_round(system_prompt: str) -> int | None:
    for p in CONTACT_ROUND_PATTERNS:
        m = p.search(system_prompt or "")
        if m:
            return int(m.group(1))
    return None


def parse_age_gender_deadline_round(system_prompt: str) -> int | None:
    rounds: list[int] = []
    for p in AGE_GENDER_ROUND_PATTERNS:
        rounds.extend(int(m.group(1)) for m in p.finditer(system_prompt or ""))
    return min(rounds) if rounds else None


def is_hard_dual_format_valid(text: str) -> bool:
    payload = (text or "").strip()
    if (
        payload.count("BEGIN_META") != 1
        or payload.count("END_META") != 1
        or payload.count("BEGIN_FINAL") != 1
        or payload.count("END_FINAL") != 1
    ):
        return False

    m = FORMAT_PATTERN.match(payload)
    if not m:
        return False

    meta_lines = [line for line in m.group("meta").split("\n") if line != ""]
    if len(meta_lines) < 2:
        return False
    if not meta_lines[0].startswith("action="):
        return False
    if not meta_lines[1].startswith("thought="):
        return False
    if not all("=" in line for line in meta_lines):
        return False
    if not all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", line) for line in meta_lines):
        return False
    if not m.group("final").strip():
        return False

    return True


def load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}")
    return data


def collect_random_candidates(source: str, path: str) -> dict[str, Any]:
    records = load_json_records(path)
    print(f"[collect] source={source} conversations={len(records)}")

    # key: conv_idx -> eligible assistant turn ids in this conversation
    by_conv: dict[int, list[tuple[int, int]]] = defaultdict(list)
    stats = Counter()

    for conv_idx, rec in enumerate(records):
        conv = rec.get("conversations") or []
        if not isinstance(conv, list):
            continue

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
            if not is_hard_dual_format_valid(content):
                continue

            stats["format_valid"] += 1
            by_conv[conv_idx].append((conv_idx, turn_idx))
            if last_user_round is not None:
                stats[f"round_{last_user_round}"] += 1
            else:
                stats["round_unknown"] += 1

    # Randomly choose one valid assistant turn per conversation.
    selected_ids: set[tuple[int, int]] = set()
    pick_stats = Counter()
    rng = random.Random(20260403)

    for conv_idx, candidates in by_conv.items():
        if not candidates:
            continue
        chosen_cid = rng.choice(candidates)
        selected_ids.add(chosen_cid)
        pick_stats["picked_conversations"] += 1

    print(
        f"[collect] source={source} selected={len(selected_ids)} "
        f"format_valid={stats['format_valid']} picked_conversations={pick_stats['picked_conversations']}"
    )

    return {
        "selected_ids": selected_ids,
        "stats": stats,
        "pick_stats": pick_stats,
    }


def build_rows_for_source(
    source: str,
    path: str,
    selected_ids: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    if not selected_ids:
        return []

    records = load_json_records(path)
    rows: list[dict[str, Any]] = []

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

            if not is_hard_dual_format_valid(content):
                normalized_history.append({"role": "assistant", "content": content})
                continue

            history_snapshot = [dict(msg) for msg in normalized_history]
            prompt_messages: list[dict[str, str]] = []
            if system_prompt.strip():
                prompt_messages.append({"role": "system", "content": system_prompt})
            prompt_messages.extend(history_snapshot)

            question = ""
            for hist_turn in reversed(normalized_history):
                if hist_turn["role"] == "user" and hist_turn["content"].strip():
                    question = hist_turn["content"].strip()
                    break

            sample_id = f"{source}_{conv_idx}_{turn_idx}"
            extra_info = {
                "sample_id": sample_id,
                "source": source,
                "conv_id": conv_idx,
                "turn_id": turn_idx,
                "slice_bucket": "random_any_round",
                "original_system_prompt": system_prompt,
                "rule_contact_round": contact_round,
                "rule_age_gender_deadline_round": age_deadline,
                "turn_round": last_user_round,
                "question": question,
                "conversations": [dict(msg) for msg in history_snapshot],
                "exclude_last_turn": False,
                "truncation_policy": "random_any_round_per_conversation",
            }

            # Reward metadata keeps simple rule-compatible payload plus supervision anchor.
            reward_model = {
                "style": "random_any_round_rule",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_balanced(rows: list[dict[str, Any]], total: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= total:
        random.Random(seed).shuffle(rows)
        return rows

    rng = random.Random(seed)
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["data_source"]].append(row)

    per_hard = total // 2
    per_normal = total - per_hard

    rng.shuffle(by_source["hard"])
    rng.shuffle(by_source["normal"])

    sampled = by_source["hard"][:per_hard] + by_source["normal"][:per_normal]
    rng.shuffle(sampled)
    return sampled


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
    prompt_char_lens = [
        sum(len(msg.get("content", "")) for msg in row.get("prompt", []))
        for row in rows
    ]
    answer_char_lens = [len(row.get("ground_truth", "")) for row in rows]

    has_system = sum(
        1
        for row in rows
        if row.get("prompt")
        and isinstance(row["prompt"], list)
        and row["prompt"][0].get("role") == "system"
    )

    conv_pairs = set()
    for row in rows:
        conv_pairs.add((row["data_source"], row["extra_info"].get("conv_id")))

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
                sum(1 for row in rows if row["extra_info"].get("rule_contact_round") is not None) / len(rows),
                4,
            )
            if rows
            else 0.0,
            "age_gender_deadline_known_ratio": round(
                sum(
                    1
                    for row in rows
                    if row["extra_info"].get("rule_age_gender_deadline_round") is not None
                )
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
    parser.add_argument("--hard-json", default=SOURCE_PATHS["hard"])
    parser.add_argument("--normal-json", default=SOURCE_PATHS["normal"])
    parser.add_argument(
        "--output-dir",
        default="/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k",
    )
    parser.add_argument("--total-samples", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {"hard": args.hard_json, "normal": args.normal_json}

    collected = {
        src: collect_random_candidates(src, source_paths[src])
        for src in SOURCES
    }

    rows_all: list[dict[str, Any]] = []
    for src in SOURCES:
        rows_all.extend(
            build_rows_for_source(
                source=src,
                path=source_paths[src],
                selected_ids=collected[src]["selected_ids"],
            )
        )

    if not rows_all:
        raise RuntimeError("No rows built. Please check source files and format constraints.")

    rows = sample_balanced(rows_all, total=args.total_samples, seed=args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = min(max(args.val_size, 0), len(rows))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    all_jsonl = output_dir / "single_turn_rl_random_rounds.all.jsonl"
    train_jsonl = output_dir / "single_turn_rl_random_rounds.train.jsonl"
    val_jsonl = output_dir / "single_turn_rl_random_rounds.val.jsonl"
    write_jsonl(all_jsonl, rows)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    all_parquet = output_dir / "single_turn_rl_random_rounds.all.parquet"
    train_parquet = output_dir / "single_turn_rl_random_rounds.train.parquet"
    val_parquet = output_dir / "single_turn_rl_random_rounds.val.parquet"
    Dataset.from_list(rows).to_parquet(str(all_parquet))
    Dataset.from_list(train_rows).to_parquet(str(train_parquet))
    Dataset.from_list(val_rows).to_parquet(str(val_parquet))

    stats = {
        "seed": args.seed,
        "targets": {
            "total_samples": args.total_samples,
            "val_size": val_size,
        },
        "selected_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "by_source": dict(Counter(row["data_source"] for row in rows)),
            "turn_round": dict(Counter(row["extra_info"]["turn_round"] for row in rows)),
        },
        "candidate_stats": {
            src: {
                "assistant_total": int(collected[src]["stats"]["assistant_total"]),
                "format_valid": int(collected[src]["stats"]["format_valid"]),
                "picked_conversations": int(collected[src]["pick_stats"]["picked_conversations"]),
                "selected_after_conv_pick": int(len(collected[src]["selected_ids"])),
            }
            for src in SOURCES
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
            "all_parquet": str(all_parquet),
            "train_parquet": str(train_parquet),
            "val_parquet": str(val_parquet),
        },
    }

    stats_path = output_dir / "single_turn_rl_random_rounds.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote dataset:")
    print(f"  all:   {all_parquet}")
    print(f"  train: {train_parquet}")
    print(f"  val:   {val_parquet}")
    print(f"  stats: {stats_path}")
    print("[stats] split rows:")
    print(
        "  all/train/val="
        f"{stats['split_stats']['all']['rows']}/"
        f"{stats['split_stats']['train']['rows']}/"
        f"{stats['split_stats']['val']['rows']}"
    )
    print("[stats] turn_round top10 (all):")
    for item in stats["split_stats"]["all"]["turn_round"]["top10"]:
        print(f"  round={item['turn_round']} count={item['count']}")


if __name__ == "__main__":
    main()
