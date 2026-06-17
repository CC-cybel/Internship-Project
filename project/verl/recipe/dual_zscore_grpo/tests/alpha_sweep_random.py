#!/usr/bin/env python
"""Random simulation for choosing Dual Z-Score GRPO alpha.

This generates synthetic prompt groups with controllable between-prompt level
variance and within-prompt rollout variance. It reports averages over many
random batches so you can see how alpha behaves under different reward shapes.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from omegaconf import OmegaConf

import recipe.dual_zscore_grpo.dual_zscore_advantage  # noqa: F401
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _simulate_once(
    *,
    num_groups: int,
    group_size: int,
    level_std: float,
    intra_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    levels = torch.clamp(0.5 + level_std * torch.randn(num_groups, generator=gen), 0.0, 1.0)
    rewards = []
    uids = []
    level_per_sample = []
    for group_idx, level in enumerate(levels):
        group_rewards = torch.clamp(level + intra_std * torch.randn(group_size, generator=gen), 0.0, 1.0)
        for reward in group_rewards:
            rewards.append([float(reward)])
            uids.append(f"g{group_idx}")
            level_per_sample.append(float(level))
    return (
        torch.tensor(rewards, dtype=torch.float32),
        torch.ones(len(rewards), 1, dtype=torch.float32),
        np.array(uids, dtype=object),
        torch.tensor(level_per_sample, dtype=torch.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--level-std", type=float, default=0.25, help="Between-prompt quality variance.")
    parser.add_argument("--intra-std", type=float, default=0.04, help="Within-prompt rollout variance.")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output-mode", choices=["raw", "tanh", "clip"], default="raw")
    parser.add_argument("--tanh-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if any(math.isnan(x) or x < 0 or x > 1 for x in alphas):
        raise SystemExit("--alphas must contain values in [0, 1]")

    fn = get_adv_estimator_fn("dual_zscore_grpo")
    rows = []
    for alpha in alphas:
        corr_reward_values = []
        corr_level_values = []
        group_sep_values = []
        abs_group_mean_values = []
        min_values = []
        max_values = []
        for trial in range(args.trials):
            rewards, mask, uids, levels = _simulate_once(
                num_groups=args.num_groups,
                group_size=args.group_size,
                level_std=args.level_std,
                intra_std=args.intra_std,
                seed=trial,
            )
            config = OmegaConf.create(
                {
                    "dual_zscore_alpha": alpha,
                    "dual_zscore_output_mode": args.output_mode,
                    "dual_zscore_tanh_scale": args.tanh_scale,
                }
            )
            adv, _ = fn(rewards, mask, uids, config=config)
            scalars = adv.squeeze(-1)
            raw_scores = rewards.squeeze(-1)

            unique_uids = np.array(sorted(set(uids.tolist())), dtype=object)
            group_means = []
            reward_group_means = []
            for uid in unique_uids:
                group_means.append(float(scalars[uids == uid].mean()))
                reward_group_means.append(float(raw_scores[uids == uid].mean()))
            best = int(torch.tensor(reward_group_means).argmax())
            worst = int(torch.tensor(reward_group_means).argmin())

            corr_reward_values.append(_pearson(scalars, raw_scores))
            corr_level_values.append(_pearson(scalars, levels))
            group_sep_values.append(group_means[best] - group_means[worst])
            abs_group_mean_values.append(float(torch.tensor(group_means).abs().mean()))
            min_values.append(float(scalars.min()))
            max_values.append(float(scalars.max()))

        rows.append(
            (
                alpha,
                np.nanmean(corr_reward_values),
                np.nanmean(corr_level_values),
                np.mean(group_sep_values),
                np.mean(abs_group_mean_values),
                np.mean(min_values),
                np.mean(max_values),
            )
        )

    print(
        "Synthetic setting: "
        f"num_groups={args.num_groups}, group_size={args.group_size}, "
        f"level_std={args.level_std}, intra_std={args.intra_std}, trials={args.trials}, "
        f"output_mode={args.output_mode}"
    )
    print("alpha  corr_reward  corr_level  sep_best_worst  mean_abs_group_adv  avg_min  avg_max")
    print("-----  -----------  ----------  --------------  ------------------  -------  -------")
    for row in rows:
        print(
            f"{row[0]:5.2f}  {row[1]:11.3f}  {row[2]:10.3f}  "
            f"{row[3]:14.3f}  {row[4]:18.3f}  {row[5]:7.3f}  {row[6]:7.3f}"
        )
    print("\nHeuristic:")
    print("  More sep_best_worst means stronger level preservation.")
    print("  More corr_reward means the final signal still follows per-response reward.")
    print("  alpha around 0.8-0.9 is usually a conservative first sweep.")


if __name__ == "__main__":
    main()

