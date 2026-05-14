#!/usr/bin/env python3
"""
Minimal interview-friendly implementations of PPO and GRPO.

Design goals:
- Keep formulas explicit and easy to hand-write on a whiteboard.
- Use numpy only (no framework dependency) for clarity.
- Separate PPO and GRPO into two classes with focused responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


EPS = 1e-8


def _clip(x: np.ndarray | float, lo: float, hi: float):
    return np.minimum(np.maximum(x, lo), hi)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0


class PPOTrainerLite:
    """PPO core equations: GAE + clipped policy loss + clipped value loss."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            rewards: shape [T]
            values: shape [T]
            dones: shape [T], 1.0 if episode terminates at step t else 0.0
            last_value: V(s_{T}) bootstrap
        Returns:
            advantages, returns
        """
        T = rewards.shape[0]
        adv = np.zeros(T, dtype=np.float32)

        gae = 0.0
        for t in range(T - 1, -1, -1):
            next_value = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * gae
            adv[t] = gae

        returns = adv + values
        return adv, returns

    def policy_loss(
        self,
        old_log_probs: np.ndarray,
        new_log_probs: np.ndarray,
        advantages: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        ratio = np.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = _clip(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages
        # PPO objective is maximized, so loss is negative.
        loss = -np.mean(np.minimum(unclipped, clipped))

        clip_frac = float(np.mean((ratio > 1.0 + self.cfg.clip_range) | (ratio < 1.0 - self.cfg.clip_range)))
        approx_kl = float(np.mean(old_log_probs - new_log_probs))
        return float(loss), {"clip_frac": clip_frac, "approx_kl": approx_kl}

    def value_loss(
        self,
        old_values: np.ndarray,
        new_values: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        v_clipped = old_values + _clip(new_values - old_values, -self.cfg.value_clip_range, self.cfg.value_clip_range)
        loss_unclipped = (new_values - returns) ** 2
        loss_clipped = (v_clipped - returns) ** 2
        return float(0.5 * np.mean(np.maximum(loss_unclipped, loss_clipped)))


@dataclass
class GRPOConfig:
    clip_range: float = 0.2
    normalize_by_std: bool = True


class GRPOTrainerLite:
    """GRPO core: group-normalized advantage + PPO-style clipped objective."""

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg

    def compute_group_advantages(self, rewards: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            rewards: shape [N], one scalar reward per sampled response
            group_ids: shape [N], samples with same id belong to one prompt group
        Returns:
            group-normalized advantages, shape [N]
        """
        adv = np.zeros_like(rewards, dtype=np.float32)
        unique_groups = np.unique(group_ids)

        for gid in unique_groups:
            idx = np.where(group_ids == gid)[0]
            r = rewards[idx]
            mu = float(np.mean(r))
            if self.cfg.normalize_by_std:
                sigma = float(np.std(r))
                adv[idx] = (r - mu) / max(sigma, EPS)
            else:
                adv[idx] = r - mu

        return adv

    def policy_loss(
        self,
        old_log_probs: np.ndarray,
        new_log_probs: np.ndarray,
        advantages: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        ratio = np.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = _clip(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages
        loss = -np.mean(np.minimum(unclipped, clipped))

        clip_frac = float(np.mean((ratio > 1.0 + self.cfg.clip_range) | (ratio < 1.0 - self.cfg.clip_range)))
        approx_kl = float(np.mean(old_log_probs - new_log_probs))
        return float(loss), {"clip_frac": clip_frac, "approx_kl": approx_kl}


if __name__ == "__main__":
    # Tiny runnable demo
    np.random.seed(7)

    # ----- PPO demo -----
    ppo = PPOTrainerLite(PPOConfig())
    T = 8
    rewards = np.random.randn(T).astype(np.float32)
    values = np.random.randn(T).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0

    adv, ret = ppo.compute_gae(rewards, values, dones, last_value=0.0)
    old_lp = np.random.randn(T).astype(np.float32)
    new_lp = old_lp + 0.05 * np.random.randn(T).astype(np.float32)

    ppo_pg_loss, ppo_info = ppo.policy_loss(old_lp, new_lp, adv)
    ppo_v_loss = ppo.value_loss(values, values + 0.1 * np.random.randn(T).astype(np.float32), ret)
    print("[PPO] policy_loss=", round(ppo_pg_loss, 6), "value_loss=", round(ppo_v_loss, 6), ppo_info)

    # ----- GRPO demo -----
    grpo = GRPOTrainerLite(GRPOConfig())
    N = 12
    group_ids = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3])
    group_rewards = np.random.randn(N).astype(np.float32)
    group_adv = grpo.compute_group_advantages(group_rewards, group_ids)

    old_lp_g = np.random.randn(N).astype(np.float32)
    new_lp_g = old_lp_g + 0.05 * np.random.randn(N).astype(np.float32)
    grpo_pg_loss, grpo_info = grpo.policy_loss(old_lp_g, new_lp_g, group_adv)
    print("[GRPO] policy_loss=", round(grpo_pg_loss, 6), grpo_info)
