#!/usr/bin/env python3
"""Download a prompt-only teacher rollout pool from public datasets.

This script:
1) samples prompts from general public datasets via Hugging Face mirror,
2) deduplicates globally,
3) removes overlaps against OPD pool prompts by normalized first-user text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass
class SourceSpec:
    name: str
    dataset: str
    target: int
    split_candidates: list[str]
    config_candidates: list[str | None]
    shuffle_buffer: int = 50_000


# General public datasets for teacher rollout prompt generation.
# Keep this set disjoint from the OPD source list used in download_opd_prompt_pool.py.
SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="openorca",
        dataset="Open-Orca/OpenOrca",
        target=30_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="slimorca",
        dataset="Open-Orca/SlimOrca",
        target=20_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="evol_instruct",
        dataset="WizardLM/WizardLM_evol_instruct_V2_196k",
        target=20_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="codealpaca",
        dataset="sahil2801/CodeAlpaca-20k",
        target=5_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="alpaca_en",
        dataset="llamafactory/alpaca_en",
        target=5_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
]


def _normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user", "prompter"}:
        return "user"
    if role in {"assistant", "gpt", "bot"}:
        return "assistant"
    return role


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def _extract_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    candidate_keys = ["messages", "conversation", "conversations", "chat", "prompt"]
    for key in candidate_keys:
        val = record.get(key)
        if not isinstance(val, list):
            continue

        messages: list[dict[str, str]] = []
        for item in val:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("from") or item.get("speaker")
            content = item.get("content") or item.get("value") or item.get("text")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            role = _normalize_role(role)
            content = content.strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})

        if not messages:
            continue
        if any(msg["role"] == "user" for msg in messages):
            return messages
    return None


def _extract_text_prompt(record: dict[str, Any]) -> str | None:
    # Prefer instruction+input style when present.
    instruction = record.get("instruction")
    input_text = record.get("input")
    if isinstance(instruction, str) and instruction.strip():
        if isinstance(input_text, str) and input_text.strip():
            merged = f"{instruction.strip()}\n\n{input_text.strip()}"
            if len(merged) >= 8:
                return merged
        if len(instruction.strip()) >= 8:
            return instruction.strip()

    text_keys = [
        "question",
        "prompt",
        "inputs",
        "query",
        "problem",
        "text",
        "input",
    ]
    for key in text_keys:
        val = record.get(key)
        if isinstance(val, str):
            # Some datasets (e.g. OASST1) store one message per row with a role field.
            # For prompt pooling, keep user-side texts only when role is explicitly provided.
            if key == "text":
                role = record.get("role")
                if isinstance(role, str) and _normalize_role(role) != "user":
                    continue
            text = val.strip()
            if len(text) >= 8:
                return text
    return None


def _extract_prompt(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = _extract_messages(record)
    if messages is not None:
        return {"type": "messages", "value": messages}

    text = _extract_text_prompt(record)
    if text is not None:
        return {"type": "text", "value": text}

    return None


def _first_user_text(prompt_obj: dict[str, Any]) -> str:
    ptype = prompt_obj.get("type")
    value = prompt_obj.get("value")
    if ptype == "text" and isinstance(value, str):
        return value
    if ptype == "messages" and isinstance(value, list):
        for m in value:
            if isinstance(m, dict) and m.get("role") == "user":
                text = m.get("content")
                if isinstance(text, str):
                    return text
    return ""


def _canonical_prompt(prompt_obj: dict[str, Any]) -> str:
    return json.dumps(prompt_obj, ensure_ascii=False, sort_keys=True)


def _first_user_hash(prompt_obj: dict[str, Any]) -> str:
    text = _normalize_text(_first_user_text(prompt_obj))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_streaming_dataset(spec: SourceSpec, seed: int):
    last_err: Exception | None = None
    for config_name in spec.config_candidates:
        for split_name in spec.split_candidates:
            try:
                ds = load_dataset(
                    spec.dataset,
                    config_name,
                    split=split_name,
                    streaming=True,
                    trust_remote_code=True,
                )
                if spec.shuffle_buffer > 0:
                    ds = ds.shuffle(seed=seed, buffer_size=spec.shuffle_buffer)
                return ds, config_name, split_name
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
    assert last_err is not None
    raise last_err


def _load_opd_exclude_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = row.get("prompt")
            prompt_obj: dict[str, Any] | None = None
            if isinstance(prompt, list):
                prompt_obj = {"type": "messages", "value": prompt}
            elif isinstance(prompt, str):
                prompt_obj = {"type": "text", "value": prompt}
            if prompt_obj is None:
                continue
            hashes.add(_first_user_hash(prompt_obj))
    return hashes


def sample_source(
    spec: SourceSpec,
    seed: int,
    writer,
    global_prompt_dedup: set[str],
    global_first_user_dedup: set[str],
    exclude_first_user_hashes: set[str],
    max_scan_multiplier: int,
) -> dict[str, Any]:
    ds, config_name, split_name = _load_streaming_dataset(spec, seed)

    target = spec.target
    max_scan = max(target * max_scan_multiplier, target + 1_000)
    accepted = 0
    scanned = 0
    skipped_by_opd = 0
    skipped_by_dup = 0

    for record in ds:
        scanned += 1
        if scanned > max_scan:
            break
        if not isinstance(record, dict):
            continue

        prompt_obj = _extract_prompt(record)
        if prompt_obj is None:
            continue

        canon = _canonical_prompt(prompt_obj)
        digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()
        if digest in global_prompt_dedup:
            skipped_by_dup += 1
            continue

        first_user_digest = _first_user_hash(prompt_obj)
        if first_user_digest in exclude_first_user_hashes:
            skipped_by_opd += 1
            continue
        if first_user_digest in global_first_user_dedup:
            skipped_by_dup += 1
            continue

        global_prompt_dedup.add(digest)
        global_first_user_dedup.add(first_user_digest)
        accepted += 1

        row = {
            "id": f"{spec.name}_{accepted:06d}",
            "data_source": spec.name,
            "dataset": spec.dataset,
            "config": config_name,
            "split": split_name,
            "prompt": prompt_obj["value"],
            "prompt_type": prompt_obj["type"],
        }
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")

        if accepted >= target:
            break

    return {
        "name": spec.name,
        "dataset": spec.dataset,
        "config": config_name,
        "split": split_name,
        "target": target,
        "accepted": accepted,
        "scanned": scanned,
        "skipped_by_opd": skipped_by_opd,
        "skipped_by_dup": skipped_by_dup,
        "status": "ok" if accepted >= target else "shortfall",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download prompt-only teacher rollout pool from public datasets.")
    parser.add_argument(
        "--output-dir",
        default="~/datasets/teacher_rollout_prompt_pool_public",
        help="Output directory for sampled files and reports.",
    )
    parser.add_argument(
        "--exclude-opd",
        default="~/datasets/opd_prompt_pool_50k/opd_prompt_pool_50k_clean.jsonl",
        help="OPD pool file for overlap removal by first-user prompt hash.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-scan-multiplier",
        type=int,
        default=120,
        help="Per-source max scanned rows = target * this value.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face mirror endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "teacher_rollout_prompt_pool_public_raw.jsonl"
    report_path = out_dir / "download_report.json"

    exclude_path = Path(args.exclude_opd).expanduser().resolve()
    exclude_hashes = _load_opd_exclude_hashes(exclude_path)

    global_prompt_dedup: set[str] = set()
    global_first_user_dedup: set[str] = set()
    report: dict[str, Any] = {
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "exclude_opd_file": str(exclude_path),
        "exclude_opd_hash_count": len(exclude_hashes),
        "output_file": str(merged_path),
        "sources": [],
        "seed": args.seed,
    }

    with merged_path.open("w", encoding="utf-8") as writer:
        for idx, spec in enumerate(SOURCES):
            try:
                source_stat = sample_source(
                    spec=spec,
                    seed=args.seed + idx,
                    writer=writer,
                    global_prompt_dedup=global_prompt_dedup,
                    global_first_user_dedup=global_first_user_dedup,
                    exclude_first_user_hashes=exclude_hashes,
                    max_scan_multiplier=args.max_scan_multiplier,
                )
            except Exception as exc:  # noqa: BLE001
                source_stat = {
                    "name": spec.name,
                    "dataset": spec.dataset,
                    "target": spec.target,
                    "accepted": 0,
                    "scanned": 0,
                    "skipped_by_opd": 0,
                    "skipped_by_dup": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            report["sources"].append(source_stat)
            print(
                f"[{source_stat['status']}] {spec.name}: "
                f"accepted={source_stat.get('accepted', 0)} / target={spec.target}, "
                f"opd_skip={source_stat.get('skipped_by_opd', 0)}"
            )

    total_target = sum(spec.target for spec in SOURCES)
    total_accepted = sum(int(item.get("accepted", 0)) for item in report["sources"])
    total_opd_skip = sum(int(item.get("skipped_by_opd", 0)) for item in report["sources"])
    total_dup_skip = sum(int(item.get("skipped_by_dup", 0)) for item in report["sources"])
    report["total_target"] = total_target
    report["total_accepted"] = total_accepted
    report["total_skipped_by_opd"] = total_opd_skip
    report["total_skipped_by_dup"] = total_dup_skip
    report["global_unique_prompt_count"] = len(global_prompt_dedup)
    report["global_unique_first_user_count"] = len(global_first_user_dedup)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n[DONE] prompts={total_accepted}/{total_target}")
    print(f"- raw jsonl: {merged_path}")
    print(f"- report:    {report_path}")
    print("If there is shortfall, rerun with a larger --max-scan-multiplier.")


if __name__ == "__main__":
    main()
