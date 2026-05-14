#!/usr/bin/env python3
"""
Prepare a single-turn RL dataset focused on contact-acquisition stage.

This script is designed for "套联阶段" training:
- For each multi-turn conversation, it picks the assistant turn around contact stage,
  prioritizing the turn at rule_contact_round with explicit contact-acquisition signal.
- The output schema is compatible with verl GRPO single-turn training.

Output fields per row:
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

CONTACT_SIGNAL_PATTERNS = (
    re.compile(r"电话|手机号|联系方式|留个(?:电话|微信)|加(?:我|下)|微信|回电|拨打|联系我|联系你"),
    re.compile(r"k\d{6,}|1[3-9]\d{9}"),
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
    return int(m.group(1)) if m else None


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


def extract_final_block(text: str) -> str:
    m = FORMAT_PATTERN.match((text or "").strip())
    if not m:
        return ""
    return m.group("final").strip()


def has_contact_signal(assistant_text: str) -> bool:
    final_text = extract_final_block(assistant_text)
    if not final_text:
        return False
    return any(p.search(final_text) for p in CONTACT_SIGNAL_PATTERNS)


def load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}")
    return data


def choose_contact_turn_ids(
    source: str,
    path: str,
    strict_contact_signal: bool,
) -> dict[str, Any]:
    records = load_json_records(path)
    print(f"[collect] source={source} conversations={len(records)}")

    selected_ids: set[tuple[int, int]] = set()
    stats = Counter()

    for conv_idx, rec in enumerate(records):
        conv = rec.get("conversations") or []
        if not isinstance(conv, list):
            continue

        system_prompt = str(rec.get("system", ""))
        contact_round = parse_contact_round(system_prompt)
        if contact_round is None:
            stats["missing_contact_round"] += 1
            continue
        if contact_round <= 1:
            stats["filtered_contact_round_le1"] += 1
            continue

        candidates: list[dict[str, Any]] = []
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
            if last_user_round is None or last_user_round <= 1:
                stats["filtered_turn_round_le1"] += 1
                continue

            signal = has_contact_signal(content)
            candidates.append(
                {
                    "conv_idx": conv_idx,
                    "turn_idx": turn_idx,
                    "round": last_user_round,
                    "signal": signal,
                }
            )

        if not candidates:
            stats["no_valid_assistant_turn"] += 1
            continue

        exact_signal = [c for c in candidates if c["round"] == contact_round and c["signal"]]
        exact_any = [c for c in candidates if c["round"] == contact_round]
        after_signal = [
            c for c in candidates if c["round"] is not None and c["round"] >= contact_round and c["signal"]
        ]

        chosen: dict[str, Any] | None = None
        if strict_contact_signal:
            if exact_signal:
                chosen = sorted(exact_signal, key=lambda x: x["turn_idx"])[0]
                stats["picked_exact_with_signal"] += 1
            else:
                stats["no_exact_contact_turn"] += 1
                continue
        else:
            if exact_signal:
                chosen = sorted(exact_signal, key=lambda x: x["turn_idx"])[0]
                stats["picked_exact_with_signal"] += 1
            elif exact_any:
                chosen = sorted(exact_any, key=lambda x: x["turn_idx"])[0]
                stats["picked_exact_no_signal"] += 1
            elif after_signal:
                chosen = sorted(after_signal, key=lambda x: (x["round"], x["turn_idx"]))[0]
                stats["picked_after_with_signal"] += 1
            else:
                stats["no_contact_stage_turn"] += 1
                continue

        selected_ids.add((chosen["conv_idx"], chosen["turn_idx"]))
        stats["picked_conversations"] += 1

    print(
        f"[collect] source={source} selected={len(selected_ids)} "
        f"picked_conversations={stats['picked_conversations']} "
        f"missing_contact_round={stats['missing_contact_round']}"
    )

    return {
        "selected_ids": selected_ids,
        "stats": stats,
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
                "slice_bucket": "contact_stage",
                "original_system_prompt": system_prompt,
                "rule_contact_round": contact_round,
                "rule_age_gender_deadline_round": age_deadline,
                "turn_round": last_user_round,
                "question": question,
                "conversations": [dict(msg) for msg in history_snapshot],
                "exclude_last_turn": False,
                "truncation_policy": "contact_stage_per_conversation",
            }

            reward_model = {
                "style": "contact_stage_rule",
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


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from datasets import Dataset  # type: ignore
    except Exception:
        return False

    Dataset.from_list(rows).to_parquet(str(path))
    return True


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

    prompt_turn_counts = [len(row.get("prompt", [])) for row in rows]
    prompt_char_lens = [sum(len(msg.get("content", "")) for msg in row.get("prompt", [])) for row in rows]
    answer_char_lens = [len(row.get("ground_truth", "")) for row in rows]

    has_system = sum(
        1
        for row in rows
        if row.get("prompt") and isinstance(row["prompt"], list) and row["prompt"][0].get("role") == "system"
    )

    conv_pairs = {(row["data_source"], row["extra_info"].get("conv_id")) for row in rows}

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
        },
        "turn_round": {str(k): int(v) for k, v in sorted(by_round.items(), key=lambda kv: str(kv[0]))},
        "rule_contact_round": {str(k): int(v) for k, v in sorted(by_contact_round.items(), key=lambda kv: str(kv[0]))},
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
        default="/data/chengch/project/rl_remake/outputs/single_turn_rl_contact_stage_2k",
    )
    parser.add_argument("--total-samples", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-contact-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default true. When enabled, only keep the exact contact_round assistant turn "
            "with explicit contact-acquisition signal."
        ),
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
            )
        )

    if not rows_all:
        raise RuntimeError("No rows built. Check source data and contact-stage extraction rules.")

    rows = sample_balanced(rows_all, total=args.total_samples, seed=args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = min(max(args.val_size, 0), len(rows))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    all_jsonl = output_dir / "single_turn_rl_contact_stage.all.jsonl"
    train_jsonl = output_dir / "single_turn_rl_contact_stage.train.jsonl"
    val_jsonl = output_dir / "single_turn_rl_contact_stage.val.jsonl"
    write_jsonl(all_jsonl, rows)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    all_parquet = output_dir / "single_turn_rl_contact_stage.all.parquet"
    train_parquet = output_dir / "single_turn_rl_contact_stage.train.parquet"
    val_parquet = output_dir / "single_turn_rl_contact_stage.val.parquet"
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
        "selected_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "by_source": dict(Counter(row["data_source"] for row in rows)),
            "turn_round": dict(Counter(row["extra_info"].get("turn_round") for row in rows)),
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

    stats_path = output_dir / "single_turn_rl_contact_stage.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote dataset:")
    if parquet_ok:
        print(f"  all:   {all_parquet}")
        print(f"  train: {train_parquet}")
        print(f"  val:   {val_parquet}")
    else:
        print("  parquet: skipped (package 'datasets' not installed)")
    print(f"  stats: {stats_path}")
    print("[stats] split rows:")
    print(
        "  all/train/val="
        f"{stats['split_stats']['all']['rows']}/"
        f"{stats['split_stats']['train']['rows']}/"
        f"{stats['split_stats']['val']['rows']}"
    )


if __name__ == "__main__":
    main()
