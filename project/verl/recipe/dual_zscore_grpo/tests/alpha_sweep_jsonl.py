#!/usr/bin/env python
"""Alpha sweep on a JSONL file containing reward scores.

This is intentionally flexible because reward logs vary. You can either provide
a group key from each JSON object, or let the script chunk records sequentially
by rollout_n.

Examples:

    python recipe/dual_zscore_grpo/tests/alpha_sweep_jsonl.py \
      --input /path/to/genrm_io.jsonl \
      --score-key score \
      --group-size 4

    python recipe/dual_zscore_grpo/tests/alpha_sweep_jsonl.py \
      --input /path/to/file.jsonl \
      --score-key reward_score \
      --group-key extra_info.sample_id
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

import recipe.dual_zscore_grpo.dual_zscore_advantage  # noqa: F401
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


def _get_path(obj: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def _try_get_score(obj: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        try:
            value = _get_path(obj, key)
        except KeyError:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    raise KeyError(f"none of score keys found: {keys}")


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--score-key", action="append", default=None)
    parser.add_argument("--group-key", default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alphas", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--output-mode", choices=["raw", "tanh", "clip"], default="raw")
    parser.add_argument("--tanh-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_keys = args.score_key or [
        "score",
        "reward_score",
        "model_judge_score",
        "ab_score_output",
        "result.score",
        "reward_model.score",
    ]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if any(math.isnan(x) or x < 0 or x > 1 for x in alphas):
        raise SystemExit("--alphas must contain values in [0, 1]")

    records = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if args.limit is not None and len(records) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                score = _try_get_score(obj, score_keys)
            except KeyError:
                continue
            if args.group_key:
                try:
                    uid = str(_get_path(obj, args.group_key))
                except KeyError:
                    continue
            else:
                uid = f"chunk_{len(records) // args.group_size}"
            records.append((score, uid))

    if not records:
        raise SystemExit("No records with usable scores were found.")

    group_counts = {}
    for _, uid in records:
        group_counts[uid] = group_counts.get(uid, 0) + 1
    valid_uids = {uid for uid, count in group_counts.items() if count > 1}
    records = [(score, uid) for score, uid in records if uid in valid_uids]
    if not records:
        raise SystemExit("No groups with at least two scored records were found.")

    rewards = torch.tensor([[score] for score, _ in records], dtype=torch.float32)
    mask = torch.ones_like(rewards)
    uids = np.array([uid for _, uid in records], dtype=object)
    raw_scores = rewards.squeeze(-1)
    levels = torch.empty_like(raw_scores)
    for uid in sorted(valid_uids):
        levels[uids == uid] = raw_scores[uids == uid].mean()

    fn = get_adv_estimator_fn("dual_zscore_grpo")
    print(
        f"Loaded {len(records)} scored records from {args.input}; "
        f"groups={len(valid_uids)}, score_keys={score_keys}, "
        f"grouping={'key ' + args.group_key if args.group_key else 'sequential chunks'}"
    )
    print("alpha  corr_reward  corr_level  sep_p90_p10  mean_abs_group_adv  min_adv  max_adv")
    print("-----  -----------  ----------  -----------  ------------------  -------  -------")
    for alpha in alphas:
        config = OmegaConf.create(
            {
                "dual_zscore_alpha": alpha,
                "dual_zscore_output_mode": args.output_mode,
                "dual_zscore_tanh_scale": args.tanh_scale,
            }
        )
        adv, _ = fn(rewards, mask, uids, config=config)
        scalars = adv.squeeze(-1)
        group_adv_means = []
        group_reward_means = []
        for uid in sorted(valid_uids):
            group_adv_means.append(float(scalars[uids == uid].mean()))
            group_reward_means.append(float(raw_scores[uids == uid].mean()))
        order = np.argsort(group_reward_means)
        p10 = order[max(0, int(0.10 * (len(order) - 1)))]
        p90 = order[min(len(order) - 1, int(0.90 * (len(order) - 1)))]
        sep = group_adv_means[p90] - group_adv_means[p10]
        print(
            f"{alpha:5.2f}  {_pearson(scalars, raw_scores):11.3f}  {_pearson(scalars, levels):10.3f}  "
            f"{sep:11.3f}  {float(torch.tensor(group_adv_means).abs().mean()):18.3f}  "
            f"{float(scalars.min()):7.3f}  {float(scalars.max()):7.3f}"
        )


if __name__ == "__main__":
    main()
