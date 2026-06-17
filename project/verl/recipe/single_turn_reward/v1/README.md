# single_turn_reward/v1

`v1/` is the earliest staged single-turn reward recipe kept in this repository. It represents an older baseline stage before the later contact/mid/rubric reward versions were consolidated.

## Purpose

Use this folder mainly for historical comparison and for understanding how the first custom reward functions and launch scripts were structured.

## Typical Contents

- reward function scripts for early stage behavior;
- reward-model wrapper scripts;
- GRPO launch scripts targeting Qwen3-style models;
- rule files or helper logic used by the first reward stage.

Some original files were superseded by later versions and may be deleted from the current GitLab snapshot if they are no longer used.

## When To Use

Prefer newer folders (`v4/`, `v5/`) for active experiments. Use `v1/` only when reproducing an older result or checking how the reward design evolved.
