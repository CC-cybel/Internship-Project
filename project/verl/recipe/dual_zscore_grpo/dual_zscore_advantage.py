"""Dual Z-Score GRPO advantage estimator.

This recipe-local estimator keeps vanilla GRPO's within-prompt ordering while
adding a prompt-level reward signal from each group's mean reward. It is
registered under ``algorithm.adv_estimator=dual_zscore_grpo`` without modifying
verl's built-in trainer or core algorithm files.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


def _config_get(config: Optional[AlgoConfig], key: str, default):
    if config is None:
        return default
    return config.get(key, default)


@register_adv_est("dual_zscore_grpo")
def compute_dual_zscore_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute level-aware GRPO advantages from outcome rewards.

    For each response, first reduce token rewards to one scalar score. Then:

    * intra term: z-score inside the prompt group, preserving GRPO's relative
      comparison among rollouts for the same prompt.
    * inter term: z-score of the prompt group's mean reward across prompt
      groups in the current PPO batch, preserving level information.
    * fusion: ``alpha * intra + (1 - alpha) * inter``.

    Optional output normalization can be enabled by config:

    * ``algorithm.dual_zscore_output_mode=raw``: no range normalization.
    * ``algorithm.dual_zscore_output_mode=tanh``: ``tanh(scale * advantage)``.
    * ``algorithm.dual_zscore_output_mode=clip``: clamp to ``[-clip, clip]``.

    The default output mode is ``tanh`` to satisfy the recipe goal of bounded
    advantages in ``[-1, 1]``.
    """
    del norm_adv_by_std_in_grpo  # Dual z-score always uses std normalization.

    alpha = float(_config_get(config, "dual_zscore_alpha", 0.8))
    alpha = min(max(alpha, 0.0), 1.0)
    eps = float(_config_get(config, "dual_zscore_epsilon", epsilon))
    output_mode = str(_config_get(config, "dual_zscore_output_mode", "tanh")).lower()
    tanh_scale = float(_config_get(config, "dual_zscore_tanh_scale", 1.0))
    clip_value = float(_config_get(config, "dual_zscore_clip", 1.0))

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        bsz = scores.shape[0]

        id2positions: dict[object, list[int]] = defaultdict(list)
        for i in range(bsz):
            id2positions[index[i]].append(i)

        group_means = []
        group_stds = {}
        group_mean_by_id = {}
        for uid, positions in id2positions.items():
            group_scores = scores[positions]
            mean = group_scores.mean()
            if len(positions) > 1:
                std = group_scores.std(unbiased=False)
            else:
                std = torch.ones((), dtype=scores.dtype, device=scores.device)
            group_mean_by_id[uid] = mean
            group_stds[uid] = std
            group_means.append(mean)

        means_tensor = torch.stack(group_means)
        if means_tensor.numel() > 1:
            inter_mean = means_tensor.mean()
            inter_std = means_tensor.std(unbiased=False)
        else:
            inter_mean = means_tensor.mean()
            inter_std = torch.ones((), dtype=scores.dtype, device=scores.device)

        fused = torch.empty_like(scores)
        for i in range(bsz):
            uid = index[i]
            intra = (scores[i] - group_mean_by_id[uid]) / (group_stds[uid] + eps)
            inter = (group_mean_by_id[uid] - inter_mean) / (inter_std + eps)
            fused[i] = alpha * intra + (1.0 - alpha) * inter

        if output_mode == "tanh":
            fused = torch.tanh(tanh_scale * fused)
        elif output_mode == "clip":
            fused = torch.clamp(fused, min=-clip_value, max=clip_value)
        elif output_mode == "raw":
            pass
        else:
            raise ValueError(
                "algorithm.dual_zscore_output_mode must be one of "
                f"'raw', 'tanh', or 'clip', got {output_mode!r}"
            )

        advantages = fused.unsqueeze(-1) * response_mask
        return advantages, advantages

