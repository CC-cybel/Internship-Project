# single_turn_reward/v5

`v5/` is the most important recent reward-recipe folder. It contains rubric-style contact reward work, offline scoring utilities, and scripts used to evaluate or train models with more explicit contact-stage rubrics.

## Purpose

This folder is used to connect rubric-based evaluation with RL reward design. It supports:

- rubric-enhanced contact-stage reward functions;
- judge/model based scoring;
- hard contact-rule configs;
- offline scoring of SFT or generated samples;
- benchmark-style contact rubric evaluation.

## Key Files

- `contact_reward_hard_config.json`: deterministic hard-rule config for contact behavior.
- `rubric_reviewer.py`: utility for reviewing and applying rubric logic.
- `evaluate_contact_rubric_bench.py`: benchmark/evaluation entrypoint.
- `offline_score_high_score_sft.py`: offline scoring for high-score SFT sample selection.
- `reward_function_stage4_contact_rubric_cloud.py`: cloud-judge rubric reward function.
- `reward_model_stage4_contact_rubric_cloud.py`: reward-model wrapper.
- `run_evaluate_contact_rubric_bench.sh`: shell wrapper for rubric benchmark runs.
- `run_grpo_single_turn_4gpu_qwen3_8b_stage4_rubric_reward.sh`: GRPO training launch script.
- `rubrics/`: versioned rubric JSONs and rubric index files.

## Data Policy

`data/`, generated rubric evaluation outputs, benchmark result JSONL files, PDFs, logs, and checkpoints are ignored. Keep only code, configs, small rubric files, and documentation in Git.

## When To Use

Use `v5/` for current contact-rubric reward work, offline scoring, or benchmark alignment with `project/contact_bench/contact_rubric_bench/`.
