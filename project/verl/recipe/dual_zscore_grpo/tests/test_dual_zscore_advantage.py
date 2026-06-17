#!/usr/bin/env python
"""Small correctness checks for the Dual Z-Score GRPO estimator."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

import recipe.dual_zscore_grpo.dual_zscore_advantage  # noqa: F401
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


def main() -> None:
    fn = get_adv_estimator_fn("dual_zscore_grpo")
    rewards = torch.tensor([[0.92], [0.94], [0.96], [0.98], [0.12], [0.14], [0.16], [0.18]])
    mask = torch.ones_like(rewards)
    uids = np.array(["good"] * 4 + ["bad"] * 4, dtype=object)
    config = OmegaConf.create({"dual_zscore_alpha": 0.8, "dual_zscore_output_mode": "raw"})

    adv, returns = fn(rewards, mask, uids, config=config)
    assert torch.equal(adv, returns)
    assert abs(float(adv.mean())) < 1e-5
    assert float(adv[:4].mean()) > float(adv[4:].mean())
    assert float(adv[3]) > float(adv[0])
    assert float(adv[7]) > float(adv[4])

    bounded_config = OmegaConf.create({"dual_zscore_alpha": 0.8, "dual_zscore_output_mode": "tanh"})
    bounded, _ = fn(rewards, mask, uids, config=bounded_config)
    assert float(bounded.max()) <= 1.0
    assert float(bounded.min()) >= -1.0

    print("dual_zscore_advantage checks passed")


if __name__ == "__main__":
    main()

