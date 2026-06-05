#!/usr/bin/env python
"""Toy alpha sweep for Dual Z-Score GRPO.

Run from the verl repo root:

    /data/chengch/.conda/envs/verl/bin/python \
      recipe/dual_zscore_grpo/tests/alpha_sweep_toy.py

The table is meant for intuition, not benchmarking. Look at:

* group_mean_adv: whether globally good prompts stay positive.
* sep_high_low: high-quality group mean advantage minus low-quality group mean
  advantage.
* corr_reward: how much the final scalar advantage still tracks raw reward.
* corr_level: how much it tracks prompt-level mean reward.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

import recipe.dual_zscore_grpo.dual_zscore_advantage  # noqa: F401
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


@dataclass(frozen=True)
class Scenario:
    name: str
    rewards: list[list[float]]
    note: str


SCENARIOS = [
    Scenario(
        name="level_gap_same_shape",
        rewards=[
            [0.92, 0.94, 0.96, 0.98],
            [0.12, 0.14, 0.16, 0.18],
            [0.52, 0.54, 0.56, 0.58],
            [0.42, 0.44, 0.46, 0.48],
        ],
        note="Same within-group spread, very different group means.",
    ),
    Scenario(
        name="all_good_nearly_flat_vs_all_bad",
        rewards=[
            [0.91, 0.92, 0.93, 0.94],
            [0.08, 0.09, 0.10, 0.11],
            [0.72, 0.73, 0.74, 0.75],
            [0.25, 0.26, 0.27, 0.28],
        ],
        note="The case you described: all-good groups and all-bad groups both have weak intra spread.",
    ),
    Scenario(
        name="mixed_quality_large_intra",
        rewards=[
            [0.20, 0.50, 0.80, 0.95],
            [0.10, 0.30, 0.60, 0.90],
            [0.40, 0.45, 0.50, 0.55],
            [0.70, 0.72, 0.74, 0.76],
        ],
        note="Large within-prompt differences; too-small alpha can overemphasize group level.",
    ),
    Scenario(
        name="single_bad_rollout_inside_good_group",
        rewards=[
            [0.30, 0.94, 0.95, 0.96],
            [0.12, 0.13, 0.14, 0.15],
            [0.55, 0.56, 0.57, 0.58],
            [0.76, 0.77, 0.78, 0.79],
        ],
        note="Checks whether a bad rollout in a good group can still receive negative advantage.",
    ),
]


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _make_batch(reward_groups: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    flat = []
    uids = []
    levels = []
    for group_idx, rewards in enumerate(reward_groups):
        mean = sum(rewards) / len(rewards)
        for reward in rewards:
            flat.append([reward])
            uids.append(f"g{group_idx}")
            levels.append(mean)
    token_level_rewards = torch.tensor(flat, dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    return token_level_rewards, response_mask, np.array(uids, dtype=object), torch.tensor(levels)


def _format_values(values: Iterable[float], width: int = 7) -> str:
    return " ".join(f"{v:{width}.3f}" for v in values)


def run_scenario(scenario: Scenario, alphas: list[float], output_mode: str, tanh_scale: float) -> None:
    fn = get_adv_estimator_fn("dual_zscore_grpo")
    rewards, mask, uids, levels = _make_batch(scenario.rewards)
    raw_scores = rewards.squeeze(-1)
    group_names = [f"g{i}" for i in range(len(scenario.rewards))]
    high_group = int(torch.tensor([sum(g) / len(g) for g in scenario.rewards]).argmax())
    low_group = int(torch.tensor([sum(g) / len(g) for g in scenario.rewards]).argmin())

    print(f"\n=== {scenario.name} ===")
    print(scenario.note)
    print("group rewards:")
    for i, group in enumerate(scenario.rewards):
        print(f"  g{i} mean={sum(group)/len(group):.3f}: {_format_values(group)}")
    print("\nalpha  group_mean_adv                         sep_high_low  corr_reward  corr_level  min_adv  max_adv")
    print("-----  -------------------------------------  ------------  -----------  ----------  -------  -------")

    for alpha in alphas:
        config = OmegaConf.create(
            {
                "dual_zscore_alpha": alpha,
                "dual_zscore_output_mode": output_mode,
                "dual_zscore_tanh_scale": tanh_scale,
            }
        )
        adv, _ = fn(rewards.clone(), mask, uids, config=config)
        scalars = adv.squeeze(-1)
        group_means = []
        for group_name in group_names:
            group_means.append(float(scalars[uids == group_name].mean()))
        sep = group_means[high_group] - group_means[low_group]
        corr_reward = _pearson(scalars, raw_scores)
        corr_level = _pearson(scalars, levels)
        print(
            f"{alpha:5.2f}  {_format_values(group_means)}  "
            f"{sep:12.3f}  {corr_reward:11.3f}  {corr_level:10.3f}  "
            f"{float(scalars.min()):7.3f}  {float(scalars.max()):7.3f}"
        )

    print("\nReading guide:")
    print("  alpha=1.0 is vanilla GRPO-like: group_mean_adv should be near zero for every group.")
    print("  lower alpha means stronger prompt-level signal; high-score groups become more positive.")
    print("  if corr_level is too high and corr_reward drops too much, alpha is probably too low.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--output-mode", choices=["raw", "tanh", "clip"], default="raw")
    parser.add_argument("--tanh-scale", type=float, default=1.0)
    parser.add_argument("--scenario", choices=[s.name for s in SCENARIOS] + ["all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if any(math.isnan(x) or x < 0 or x > 1 for x in alphas):
        raise SystemExit("--alphas must contain values in [0, 1]")

    scenarios = SCENARIOS if args.scenario == "all" else [s for s in SCENARIOS if s.name == args.scenario]
    print(f"output_mode={args.output_mode}, tanh_scale={args.tanh_scale}")
    for scenario in scenarios:
        run_scenario(scenario, alphas, args.output_mode, args.tanh_scale)


if __name__ == "__main__":
    main()
