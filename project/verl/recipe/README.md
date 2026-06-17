# verl/recipe

`recipe/` contains project-specific training and distillation recipes built on top of the local verl framework. These folders are the main entrypoints for experiments; the surrounding `verl/` package provides the underlying trainer, workers, rollout engines, and distributed runtime.

## Subfolders

### `single_turn_reward/`

Versioned GRPO reward recipes for single-turn consultation behavior. This is the main RL training area for contact-stage, mid-stage, rubric reward, and related reward-model experiments.

### `opd_multi_teacher/`

On-policy distillation / multi-teacher experiments. Samples are routed to different teachers, and student learning is driven by teacher top-k distributions or KL-style objectives.

### `dual_zscore_grpo/`

Experimental advantage-normalization implementation. It adds a dual-z-score GRPO estimator and test scripts without modifying the entire training stack.

## Common Pattern

Each recipe usually contains:

- one or more `run_*.sh` launch scripts;
- reward functions or reward wrappers;
- data builder scripts;
- optional config files;
- local README notes explaining the experiment.

## Output Policy

Do not commit checkpoints, rollout logs, SwanLab/W&B logs, TensorBoard logs, or generated datasets. Keep only source code, config, rubrics, and documentation in Git.
