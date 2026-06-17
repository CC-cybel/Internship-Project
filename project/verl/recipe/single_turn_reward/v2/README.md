# single_turn_reward/v2

`v2/` is a second-stage single-turn reward recipe. It sits between the earliest baseline and the later contact-stage reward versions.

## Purpose

This folder is mainly retained for experiment lineage. It shows how the reward function, cloud judge calls, and launch parameters changed after the first version.

## Typical Contents

- stage-specific reward function scripts;
- rule-only reward alternatives;
- reward-model wrappers;
- Qwen3 GRPO launch scripts;
- small helper files used during stage migration.

## When To Use

Use only for reproducing older stage-2 experiments or comparing reward logic across versions. For active contact-stage or rubric reward training, prefer `v4/` and `v5/`.
