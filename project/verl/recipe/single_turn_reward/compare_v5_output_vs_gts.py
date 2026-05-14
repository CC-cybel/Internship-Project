#!/usr/bin/env python3
"""
对比 v5 训练后的回答（genrm_io output）与原始回答（ground_truth）。

输出格式（JSONL）：
  question, output, gts

其中：
  - question: 来自 genrm_io.question（用户当前问题）
  - output:   来自 genrm_io.output（训练后模型回答）
  - gts:      来自 parquet.ground_truth（原始参考答案）
"""

import json
import pandas as pd


def main():
    GENRM_PATH = "/data1/chengch/verl_outputs/grpo_single_turn/qwen3_8b_4gpu_stage5_mid_reward_20260417_101011/genrm_io.jsonl"
    PARQUET_PATH = "/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k_mid_stage/single_turn_rl_random_rounds_mid_stage.train.parquet"
    OUTPUT_PATH = "/data/chengch/project/rl_remake/outputs/short_v5_output_vs_gts.jsonl"

    # 1. 读取 genrm_io 后 100 条
    with open(GENRM_PATH) as f:
        all_lines = f.readlines()
    genrm_records = [json.loads(l) for l in all_lines[-100:]]
    print(f"genrm_io 总条数: {len(all_lines)}, 取后 100 条")

    # 2. 构建 question -> ground_truth 映射
    df = pd.read_parquet(PARQUET_PATH)
    gt_map: dict[str, str] = {}
    for _, row in df.iterrows():
        ei = row.get("extra_info", {})
        if isinstance(ei, dict):
            q = ei.get("question", "")
            gt = row.get("ground_truth", "")
            if q and q not in gt_map:
                gt_map[q] = gt
    print(f"parquet gt_map 条数: {len(gt_map)}")

    # 3. 匹配并输出
    matched, unmatched = 0, 0
    out_records = []
    for r in genrm_records:
        q = r.get("question", "")
        if q in gt_map:
            matched += 1
            out_records.append({
                "question": q,
                "output": r.get("output", ""),
                "gts": gt_map[q],
            })
        else:
            unmatched += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"匹配成功: {matched}, 未匹配: {unmatched}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
