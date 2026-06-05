#!/usr/bin/env python3
"""
筛选mid阶段样本并生成训练/验证parquet文件。

阶段判定：
  - start: turn_round <= 2
  - contact: contact_round > 0 and turn_round >= contact_round
  - mid: 其他情况（第3轮到留联触发前一轮）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _infer_stage(turn_round: int, contact_round: int) -> str:
    if turn_round <= 2:
        return "start"
    if contact_round > 0 and turn_round >= contact_round:
        return "contact"
    return "mid"


def _get_int(d: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if not d:
        return default
    val = d.get(key)
    if val is None:
        # 尝试其他字段名
        if key == "contact_round":
            for alt in ("rule_contact_round", "留联轮次"):
                val = d.get(alt)
                if val is not None:
                    break
    try:
        return int(val)
    except Exception:
        return default


def process_jsonl_to_mid_parquet(
    input_jsonl: Path,
    output_parquet: Path,
    max_history_turns: int = 8,
) -> dict[str, Any]:
    """处理单个jsonl文件，筛选mid阶段样本"""
    stats = {
        "total_rows": 0,
        "mid_stage_rows": 0,
        "start_stage_rows": 0,
        "contact_stage_rows": 0,
        "skipped_no_turns": 0,
        "skipped_no_ground_truth": 0,
    }

    records = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total_rows"] += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}")
                continue

            # 提取阶段信息
            extra_info = obj.get("extra_info") or {}
            turn_round = _get_int(extra_info, "turn_round")
            contact_round = _get_int(extra_info, "contact_round")

            stage = _infer_stage(turn_round, contact_round)

            if stage == "start":
                stats["start_stage_rows"] += 1
                continue
            elif stage == "contact":
                stats["contact_stage_rows"] += 1
                continue
            else:
                stats["mid_stage_rows"] += 1

            # 提取对话
            conversations = obj.get("conversations") or obj.get("history") or []
            if not conversations:
                stats["skipped_no_turns"] += 1
                continue

            # 提取ground_truth（当前轮次正确答案）
            ground_truth = (obj.get("ground_truth") or "").strip()
            if not ground_truth:
                stats["skipped_no_ground_truth"] += 1
                continue

            # 系统提示（如果有）
            system_prompt = ""
            first_role = conversations[0].get("role", "").lower() if conversations else ""
            if first_role == "system":
                system_prompt = (conversations[0].get("content") or "").strip()
                conversations = conversations[1:]

            # 构建prompt messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 保留最后N轮
            for turn in conversations[-max_history_turns:]:
                role = (turn.get("role") or turn.get("from", "")).lower()
                content = (turn.get("content") or turn.get("value", "")).strip()

                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

            record = {
                "prompt": messages,
                "ground_truth": ground_truth,
                "extra_info": json.dumps({
                    "turn_round": turn_round,
                    "contact_round": contact_round,
                    "sample_id": extra_info.get("sample_id", ""),
                    "source": extra_info.get("source", ""),
                    "conv_id": extra_info.get("conv_id", 0),
                    "turn_id": extra_info.get("turn_id", 0),
                    "stage": "mid",
                }, ensure_ascii=False),
            }
            records.append(record)

    if not records:
        raise ValueError("No mid-stage samples found!")

    # 写入parquet
    schema = pa.schema([
        pa.field("prompt", pa.list_(
            pa.struct([
                pa.field("role", pa.string()),
                pa.field("content", pa.string()),
            ])
        )),
        pa.field("ground_truth", pa.string()),
        pa.field("extra_info", pa.string()),
    ])

    prompt_arr = pa.array([r["prompt"] for r in records], type=schema[0].type)
    gt_arr = pa.array([r["ground_truth"] for r in records], type=pa.string())
    extra_arr = pa.array([r["extra_info"] for r in records], type=pa.string())

    table = pa.Table.from_arrays([prompt_arr, gt_arr, extra_arr], schema=schema)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet)

    stats["output_rows"] = len(records)
    return stats


def split_train_val(
    train_parquet: Path,
    val_parquet: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """从train parquet中划分验证集"""
    if val_ratio <= 0:
        return {"train_rows": 0, "val_rows": 0}

    table = pq.read_table(train_parquet)
    total = len(table)

    val_size = max(1, int(total * val_ratio))
    val_size = min(val_size, total - 1)

    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)

    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    train_table = table.take(list(train_indices))
    val_table = table.take(list(val_indices))

    val_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(train_table, train_parquet)
    pq.write_table(val_table, val_parquet)

    return {"train_rows": len(train_table), "val_rows": len(val_table)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True,
                        help="输入jsonl文件路径（单轮或多轮对话数据）")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出目录")
    parser.add_argument("--train-name", type=str, default="train.parquet",
                        help="训练集文件名")
    parser.add_argument("--val-name", type=str, default="val.parquet",
                        help="验证集文件名")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-turns", type=int, default=8,
                        help="保留的最大历史轮次数")
    args = parser.parse_args()

    output_dir = args.output_dir
    train_parquet = output_dir / args.train_name
    val_parquet = output_dir / args.val_name

    # Step 1: 筛选mid阶段样本
    print(f"[INFO] Processing {args.input_jsonl} ...")
    stats = process_jsonl_to_mid_parquet(
        args.input_jsonl,
        train_parquet,
        max_history_turns=args.max_history_turns,
    )

    print(f"\n=== Stage Distribution ===")
    print(f"  Total rows         : {stats['total_rows']}")
    print(f"  Start (skipped)    : {stats['start_stage_rows']}")
    print(f"  Mid (kept)         : {stats['mid_stage_rows']}")
    print(f"  Contact (skipped)  : {stats['contact_stage_rows']}")
    print(f"  Skipped (no turns) : {stats['skipped_no_turns']}")
    print(f"  Skipped (no GT)    : {stats['skipped_no_ground_truth']}")
    print(f"  Output rows        : {stats['output_rows']}")
    print(f"\n[INFO] Wrote train parquet: {train_parquet}")

    # Step 2: 划分验证集
    if args.val_ratio > 0:
        split_stats = split_train_val(train_parquet, val_parquet, args.val_ratio, args.seed)
        print(f"\n[INFO] Split into train: {split_stats['train_rows']}, val: {split_stats['val_rows']}")
        print(f"[INFO] Wrote val parquet: {val_parquet}")


if __name__ == "__main__":
    main()
