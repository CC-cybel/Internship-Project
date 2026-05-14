#!/usr/bin/env python3
"""Download and sample a 50k OPD prompt pool from open datasets via HF mirror.

Output format is JSONL and intentionally lightweight for a second-stage conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="tulu3_mix",
        dataset="allenai/tulu-3-sft-mixture",
        target=20_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="no_robots",
        dataset="HuggingFaceH4/no_robots",
        target=8_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="ultrachat_200k",
        dataset="HuggingFaceH4/ultrachat_200k",
        target=7_000,
        split_candidates=["train_sft", "train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="wildchat_1m",
        dataset="allenai/WildChat-1M",
        target=5_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="openr1_math",
        dataset="open-r1/OpenR1-Math-220k",
        target=6_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="numina_tir",
        dataset="AI-MO/NuminaMath-TIR",
        target=3_000,
        split_candidates=["train"],
        config_candidates=[None],
    ),
    SourceSpec(
        name="gsm8k",
        dataset="openai/gsm8k",
        target=1_000,
        split_candidates=["train"],
        config_candidates=["main"],
    ),
]


def _normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt", "bot"}:
        return "assistant"
    return role


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

        while messages and messages[-1]["role"] == "assistant":
            messages.pop()

        if any(msg["role"] == "user" for msg in messages):
            return messages

    return None


def _extract_text_prompt(record: dict[str, Any]) -> str | None:
    text_keys = ["prompt", "question", "instruction", "problem", "query", "input"]
    for key in text_keys:
        val = record.get(key)
        if isinstance(val, str):
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


def _canonical_prompt(prompt_obj: dict[str, Any]) -> str:
    return json.dumps(prompt_obj, ensure_ascii=False, sort_keys=True)


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


def sample_source(
    spec: SourceSpec,
    seed: int,
    writer,
    global_dedup: set[str],
    max_scan_multiplier: int,
) -> dict[str, Any]:
    ds, config_name, split_name = _load_streaming_dataset(spec, seed)

    target = spec.target
    max_scan = max(target * max_scan_multiplier, target + 1_000)
    accepted = 0
    scanned = 0

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
        if digest in global_dedup:
            continue

        global_dedup.add(digest)
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
        "status": "ok" if accepted >= target else "shortfall",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and sample 50k OPD prompt pool via HF mirror.")
    parser.add_argument(
        "--output-dir",
        default="~/datasets/opd_prompt_pool_50k",
        help="Output directory for sampled files and report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-scan-multiplier",
        type=int,
        default=80,
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

    merged_path = out_dir / "opd_prompt_pool_50k_raw.jsonl"
    report_path = out_dir / "download_report.json"

    global_dedup: set[str] = set()
    report: dict[str, Any] = {
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
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
                    global_dedup=global_dedup,
                    max_scan_multiplier=args.max_scan_multiplier,
                )
            except Exception as exc:  # noqa: BLE001
                source_stat = {
                    "name": spec.name,
                    "dataset": spec.dataset,
                    "target": spec.target,
                    "accepted": 0,
                    "scanned": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            report["sources"].append(source_stat)
            print(
                f"[{source_stat['status']}] {spec.name}: "
                f"accepted={source_stat.get('accepted', 0)} / target={spec.target}"
            )

    total_target = sum(spec.target for spec in SOURCES)
    total_accepted = sum(int(item.get("accepted", 0)) for item in report["sources"])
    report["total_target"] = total_target
    report["total_accepted"] = total_accepted
    report["global_unique_count"] = len(global_dedup)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n[DONE] prompts={total_accepted}/{total_target}")
    print(f"- raw jsonl: {merged_path}")
    print(f"- report:    {report_path}")
    print("If some sources failed/shortfall, rerun with a larger --max-scan-multiplier.")


if __name__ == "__main__":
    main()

