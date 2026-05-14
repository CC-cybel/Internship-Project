#!/usr/bin/env python3
"""
Prepare a 20k single-turn RL dataset for verl from dual SFT JSON files.

Design:
1) 10k random samples
2) 10k rule-targeted samples from system prompt constraints:
   - 5k contact-round samples
   - 5k age/gender-round samples

Output rows are verl RL-compatible fields:
- prompt: list[{"role": "...", "content": "..."}]
- ground_truth: str
- extra_info: dict
- reward_model: {"style": "rule", "ground_truth": ...}
- data_source: str
- index: str
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset


SOURCE_PATHS = {
    "hard": "/data/wangpf/project/LlamaFactory/data/hard_dual_s5_full.json",
    "normal": "/data/wangpf/project/data_remake/outputs/normal/normal_s5_dual_full_drop_system_keywords.json",
}
SOURCES = ("hard", "normal")

CONTACT_ROUND_PATTERNS = (
    re.compile(r"留联触发[^\n。]*?第\s*(\d+)\s*轮"),
    re.compile(r"强制启动首次留联[^\n。]*?第\s*(\d+)\s*轮"),
)
AGE_GENDER_ROUND_PATTERNS = (
    re.compile(r"第\s*(\d+)\s*轮前[^\n。]*?(?:询问|获取)[^\n。]*?年龄[^\n。]*?(?:性别|男女)"),
    re.compile(r"第\s*(\d+)\s*轮前[^\n。]*?(?:询问|获取)[^\n。]*?(?:性别|男女)[^\n。]*?年龄"),
)
TURN_ROUND_PATTERN = re.compile(r"当前第\s*(\d+)\s*轮")
FORMAT_PATTERN = re.compile(
    r"^BEGIN_META\n(?P<meta>[\s\S]*?)\nEND_META\nBEGIN_FINAL\n(?P<final>[\s\S]*?)\nEND_FINAL$",
    flags=re.DOTALL,
)
FINAL_BLOCK_PATTERN = re.compile(r"BEGIN_FINAL\s*(.*?)\s*END_FINAL", flags=re.DOTALL)
AGE_GENDER_KEYWORDS = (
    "年龄",
    "多大",
    "几岁",
    "性别",
    "男",
    "女",
    "男性",
    "女性",
    "男孩",
    "女孩",
    "男士",
    "女士",
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


def parse_contact_round(system_prompt: str) -> int | None:
    for pattern in CONTACT_ROUND_PATTERNS:
        match = pattern.search(system_prompt)
        if match:
            return int(match.group(1))
    return None


def parse_age_gender_deadline_round(system_prompt: str) -> int | None:
    rounds: list[int] = []
    for pattern in AGE_GENDER_ROUND_PATTERNS:
        rounds.extend(int(m.group(1)) for m in pattern.finditer(system_prompt))
    return min(rounds) if rounds else None


def parse_turn_round(text: str) -> int | None:
    match = TURN_ROUND_PATTERN.search(text)
    if match:
        return int(match.group(1))
    return None


def extract_final_text(text: str) -> str:
    match = FINAL_BLOCK_PATTERN.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def is_hard_dual_format_valid(text: str) -> bool:
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
    meta_lines = [line for line in match.group("meta").split("\n") if line != ""]
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
    if not match.group("final").strip():
        return False
    return True


def load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at {path}")
    return data


def collect_candidate_ids(source: str, path: str) -> dict[str, Any]:
    print(f"[collect] source={source} loading {path}")
    records = load_json_records(path)
    print(f"[collect] source={source} conversations={len(records)}")

    all_ids: list[tuple[int, int]] = []
    contact_strict: list[tuple[int, int]] = []
    contact_relaxed: list[tuple[int, int]] = []
    age_strict: list[tuple[int, int]] = []
    age_relaxed: list[tuple[int, int]] = []
    stats = Counter()

    for conv_idx, rec in enumerate(records):
        conv = rec.get("conversations") or []
        if not isinstance(conv, list):
            continue
        system_prompt = str(rec.get("system", ""))
        contact_round = parse_contact_round(system_prompt)
        age_deadline = parse_age_gender_deadline_round(system_prompt)
        last_user_round: int | None = None

        for turn_idx, turn in enumerate(conv):
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("from", turn.get("role")))
            content = str(turn.get("value", turn.get("content", "")))

            if role == "user":
                parsed_round = parse_turn_round(content)
                if parsed_round is not None:
                    last_user_round = parsed_round
                continue

            if role != "assistant":
                continue

            stats["assistant_total"] += 1
            if not is_hard_dual_format_valid(content):
                continue

            stats["format_valid"] += 1
            cid = (conv_idx, turn_idx)
            all_ids.append(cid)

            if contact_round is not None and last_user_round == contact_round:
                contact_strict.append(cid)
                stats["contact_strict"] += 1
            if (
                contact_round is not None
                and last_user_round is not None
                and abs(last_user_round - contact_round) <= 1
            ):
                contact_relaxed.append(cid)
                stats["contact_relaxed"] += 1

            final_text = extract_final_text(content)
            has_age_gender_kw = any(keyword in final_text for keyword in AGE_GENDER_KEYWORDS)
            if (
                age_deadline is not None
                and last_user_round is not None
                and last_user_round <= age_deadline
                and has_age_gender_kw
            ):
                age_strict.append(cid)
                stats["age_strict"] += 1
            if (
                age_deadline is not None
                and last_user_round is not None
                and last_user_round <= age_deadline + 1
                and has_age_gender_kw
            ):
                age_relaxed.append(cid)
                stats["age_relaxed"] += 1

    print(
        "[collect] source="
        f"{source} valid={stats['format_valid']} contact_strict={stats['contact_strict']} age_strict={stats['age_strict']}"
    )
    return {
        "all_ids": all_ids,
        "contact_strict": contact_strict,
        "contact_relaxed": contact_relaxed,
        "age_strict": age_strict,
        "age_relaxed": age_relaxed,
        "stats": stats,
    }


def _sample_from_levels(
    levels: Iterable[list[tuple[int, int]]],
    need: int,
    used: set[tuple[str, int, int]],
    source: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    for level in levels:
        pool = [
            cid
            for cid in level
            if (source, cid[0], cid[1]) not in used and cid not in selected_set
        ]
        rng.shuffle(pool)
        take = min(need - len(selected), len(pool))
        if take > 0:
            selected.extend(pool[:take])
            selected_set.update(pool[:take])
        if len(selected) >= need:
            break
    return selected


def select_bucket(
    *,
    name: str,
    target: int,
    levels_by_source: dict[str, list[list[tuple[int, int]]]],
    used: set[tuple[str, int, int]],
    rng: random.Random,
) -> dict[str, list[tuple[int, int]]]:
    source_targets = {"hard": target // 2, "normal": target - (target // 2)}
    selected = {src: [] for src in SOURCES}

    for src in SOURCES:
        selected[src] = _sample_from_levels(
            levels=levels_by_source[src],
            need=source_targets[src],
            used=used,
            source=src,
            rng=rng,
        )

    total_selected = len(selected["hard"]) + len(selected["normal"])
    if total_selected < target:
        need_more = target - total_selected
        combined: list[tuple[str, int, int]] = []
        for src in SOURCES:
            already = set(selected[src])
            for level in levels_by_source[src]:
                for cid in level:
                    if (src, cid[0], cid[1]) in used:
                        continue
                    if cid in already:
                        continue
                    combined.append((src, cid[0], cid[1]))
        deduped = list(dict.fromkeys(combined))
        rng.shuffle(deduped)
        for src, conv_idx, turn_idx in deduped[:need_more]:
            selected[src].append((conv_idx, turn_idx))

    for src in SOURCES:
        for conv_idx, turn_idx in selected[src]:
            used.add((src, conv_idx, turn_idx))

    print(
        f"[select] {name}: target={target} got={len(selected['hard']) + len(selected['normal'])} "
        f"(hard={len(selected['hard'])}, normal={len(selected['normal'])})"
    )
    return selected


def flatten_selected(
    bucket_name: str,
    selected: dict[str, list[tuple[int, int]]],
    bucket_map: dict[tuple[str, int, int], str],
) -> None:
    for src in SOURCES:
        for conv_idx, turn_idx in selected[src]:
            bucket_map[(src, conv_idx, turn_idx)] = bucket_name


def build_rows_for_source(
    source: str,
    path: str,
    selected_for_source: set[tuple[int, int]],
    bucket_map: dict[tuple[str, int, int], str],
) -> list[dict[str, Any]]:
    if not selected_for_source:
        return []

    print(f"[build] source={source} selected={len(selected_for_source)}")
    rows: list[dict[str, Any]] = []
    records = load_json_records(path)

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
                parsed_round = parse_turn_round(content)
                if parsed_round is not None:
                    last_user_round = parsed_round
                continue

            if role != "assistant":
                continue

            cid = (conv_idx, turn_idx)
            if cid not in selected_for_source:
                normalized_history.append({"role": "assistant", "content": content})
                continue

            if not is_hard_dual_format_valid(content):
                normalized_history.append({"role": "assistant", "content": content})
                continue

            history_snapshot = [dict(turn_obj) for turn_obj in normalized_history]
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
            bucket = bucket_map.get((source, conv_idx, turn_idx), "random")
            extra_info = {
                "sample_id": sample_id,
                "source": source,
                "conv_id": conv_idx,
                "turn_id": turn_idx,
                "slice_bucket": bucket,
                "original_system_prompt": system_prompt,
                "rule_contact_round": contact_round,
                "rule_age_gender_deadline_round": age_deadline,
                "turn_round": last_user_round,
                "question": question,
                "conversations": [dict(turn_obj) for turn_obj in history_snapshot],
                "exclude_last_turn": False,
            }
            rows.append(
                {
                    "prompt": prompt_messages,
                    "ground_truth": content,
                    "extra_info": extra_info,
                    "reward_model": {"style": "rule", "ground_truth": content},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-json", default=SOURCE_PATHS["hard"])
    parser.add_argument("--normal-json", default=SOURCE_PATHS["normal"])
    parser.add_argument("--output-dir", default="/data/wangpf/project/data_remake/outputs/single_turn_rl_20k")
    parser.add_argument("--total-samples", type=int, default=20000)
    parser.add_argument("--rule-samples", type=int, default=10000)
    parser.add_argument("--contact-samples", type=int, default=5000)
    parser.add_argument("--age-gender-samples", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rule_samples != args.contact_samples + args.age_gender_samples:
        raise ValueError("rule-samples must equal contact-samples + age-gender-samples")
    if args.total_samples < args.rule_samples:
        raise ValueError("total-samples must be >= rule-samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {"hard": args.hard_json, "normal": args.normal_json}
    collected = {src: collect_candidate_ids(src, path) for src, path in source_paths.items()}

    used: set[tuple[str, int, int]] = set()
    bucket_map: dict[tuple[str, int, int], str] = {}
    rng = random.Random(args.seed)

    contact_selected = select_bucket(
        name="contact_round",
        target=args.contact_samples,
        levels_by_source={
            "hard": [collected["hard"]["contact_strict"], collected["hard"]["contact_relaxed"]],
            "normal": [collected["normal"]["contact_strict"], collected["normal"]["contact_relaxed"]],
        },
        used=used,
        rng=rng,
    )
    flatten_selected("contact_round", contact_selected, bucket_map)

    age_selected = select_bucket(
        name="age_gender_round",
        target=args.age_gender_samples,
        levels_by_source={
            "hard": [collected["hard"]["age_strict"], collected["hard"]["age_relaxed"]],
            "normal": [collected["normal"]["age_strict"], collected["normal"]["age_relaxed"]],
        },
        used=used,
        rng=rng,
    )
    flatten_selected("age_gender_round", age_selected, bucket_map)

    random_target = args.total_samples - args.rule_samples
    random_selected = select_bucket(
        name="random",
        target=random_target,
        levels_by_source={
            "hard": [collected["hard"]["all_ids"]],
            "normal": [collected["normal"]["all_ids"]],
        },
        used=used,
        rng=rng,
    )
    flatten_selected("random", random_selected, bucket_map)

    selected_ids_by_source = {
        src: {(conv_idx, turn_idx) for s, conv_idx, turn_idx in bucket_map if s == src}
        for src in SOURCES
    }
    rows: list[dict[str, Any]] = []
    for src in SOURCES:
        rows.extend(
            build_rows_for_source(
                source=src,
                path=source_paths[src],
                selected_for_source=selected_ids_by_source[src],
                bucket_map=bucket_map,
            )
        )

    if len(rows) < args.total_samples:
        raise RuntimeError(
            f"Built rows {len(rows)} < target {args.total_samples}. "
            "Try increasing relaxed selection ranges."
        )
    if len(rows) > args.total_samples:
        rng.shuffle(rows)
        rows = rows[: args.total_samples]

    rng.shuffle(rows)
    val_size = min(max(args.val_size, 0), len(rows))
    train_rows = rows[val_size:]
    val_rows = rows[:val_size]

    all_jsonl = output_dir / "single_turn_rl_20k.all.jsonl"
    train_jsonl = output_dir / "single_turn_rl_20k.train.jsonl"
    val_jsonl = output_dir / "single_turn_rl_20k.val.jsonl"
    write_jsonl(all_jsonl, rows)
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    all_parquet = output_dir / "single_turn_rl_20k.all.parquet"
    train_parquet = output_dir / "single_turn_rl_20k.train.parquet"
    val_parquet = output_dir / "single_turn_rl_20k.val.parquet"
    Dataset.from_list(rows).to_parquet(str(all_parquet))
    Dataset.from_list(train_rows).to_parquet(str(train_parquet))
    Dataset.from_list(val_rows).to_parquet(str(val_parquet))

    stats = {
        "seed": args.seed,
        "targets": {
            "total_samples": args.total_samples,
            "rule_samples": args.rule_samples,
            "contact_samples": args.contact_samples,
            "age_gender_samples": args.age_gender_samples,
            "random_samples": random_target,
            "val_size": val_size,
        },
        "selected_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "by_source": dict(Counter(row["data_source"] for row in rows)),
            "by_bucket": dict(Counter(row["extra_info"]["slice_bucket"] for row in rows)),
            "by_source_bucket": {
                f"{src}|{bucket}": count
                for (src, bucket), count in Counter(
                    (row["data_source"], row["extra_info"]["slice_bucket"]) for row in rows
                ).items()
            },
        },
        "candidate_stats": {
            src: {
                "assistant_total": int(collected[src]["stats"]["assistant_total"]),
                "format_valid": int(collected[src]["stats"]["format_valid"]),
                "contact_strict": int(collected[src]["stats"]["contact_strict"]),
                "contact_relaxed": int(collected[src]["stats"]["contact_relaxed"]),
                "age_strict": int(collected[src]["stats"]["age_strict"]),
                "age_relaxed": int(collected[src]["stats"]["age_relaxed"]),
            }
            for src in SOURCES
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
    stats_path = output_dir / "single_turn_rl_20k.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote dataset:")
    print(f"  all:   {all_parquet}")
    print(f"  train: {train_parquet}")
    print(f"  val:   {val_parquet}")
    print(f"  stats: {stats_path}")


if __name__ == "__main__":
    main()
