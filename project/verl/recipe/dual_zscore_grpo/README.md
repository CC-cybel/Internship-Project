# Dual Z-Score GRPO

Recipe-local GRPO extension that keeps the normal verl framework untouched.

Use:

```bash
algorithm.adv_estimator=dual_zscore_grpo
+algorithm.dual_zscore_alpha=0.8
+algorithm.dual_zscore_output_mode=tanh
+algorithm.dual_zscore_tanh_scale=1.0
```

The estimator uses the existing GRPO grouping key `uid`, so it is independent
from PPO mini-batch and micro-batch settings. It computes inter-group level
information over the current PPO train batch.

Alpha intuition scripts:

```bash
/data/chengch/.conda/envs/verl/bin/python recipe/dual_zscore_grpo/tests/alpha_sweep_toy.py
/data/chengch/.conda/envs/verl/bin/python recipe/dual_zscore_grpo/tests/alpha_sweep_random.py
```

Run quick checks:

```bash
bash recipe/dual_zscore_grpo/tests/run_alpha_tests.sh
```
