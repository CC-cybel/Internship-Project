#!/usr/bin/env python3
"""
统计 jsonl 文件中所有 BEGIN_FINAL 到 END_FINAL 之间的字符长度分布。
"""

import json
import re
from collections import Counter

INPUT_FILE = "/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k_mid_stage/single_turn_rl_random_rounds_mid_stage.train.jsonl"

def extract_final(text: str) -> str | None:
    """从 assistant 消息中提取 BEGIN_FINAL...END_FINAL 之间的内容。"""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    match = re.search(r'BEGIN_FINAL\s*\n(.*?)\s*(?:\nEND_FINAL|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def main():
    lengths = []
    missing = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            data = json.loads(line)
            ground_truth = data.get("ground_truth", "")
            final_text = extract_final(ground_truth)
            if final_text is None:
                missing += 1
                print(f"  [line {line_no}] 无法提取 BEGIN_FINAL: {ground_truth[:200]!r}")
            else:
                lengths.append(len(final_text))

    if not lengths:
        print("没有找到任何有效的 BEGIN_FINAL 内容。")
        return

    import numpy as np
    lengths_arr = np.array(lengths, dtype=int)

    print("=" * 70)
    print(f"总行数: {len(lengths) + missing}")
    print(f"成功提取 BEGIN_FINAL 数: {len(lengths)}")
    print(f"提取失败数: {missing}")
    print("=" * 70)
    print(f"字符数  样本数")
    print("-" * 70)

    bins = list(range(0, 601, 50)) + [float("inf")]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = int(np.sum((lengths_arr >= lo) & (lengths_arr < hi)))
        if hi == float("inf"):
            label = f"  >{lo}"
        else:
            label = f"{lo:4d}-{hi - 1:4d}"
        bar = "#" * (count // 10)
        print(f"  {label}: {count:6d}  {bar}")

    print("-" * 70)
    print(f"统计量:")
    print(f"  最小值: {lengths_arr.min()}")
    print(f"  最大值: {lengths_arr.max()}")
    print(f"  均值:   {lengths_arr.mean():.2f}")
    print(f"  中位数: {np.median(lengths_arr):.2f}")
    print(f"  标准差: {lengths_arr.std():.2f}")

    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n分位数:")
    for p in percentiles:
        val = int(np.percentile(lengths_arr, p))
        print(f"  P{p:2d}: {val}")

    print(f"\n长度分布 Counter (每10字符):")
    counter = Counter()
    for L in lengths_arr:
        bucket = (L // 10) * 10
        counter[bucket] += 1
    for L in sorted(counter):
        bar = "#" * (counter[L] // 10)
        print(f"  {L:4d}-{L + 9:4d}: {counter[L]:6d}  {bar}")


if __name__ == "__main__":
    main()
