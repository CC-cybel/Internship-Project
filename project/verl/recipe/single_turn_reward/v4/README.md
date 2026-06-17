# single_turn_reward/v4

`v4/` is the contact-stage reward recipe area. It focuses on training a model to collect contact information at the right moment while preserving natural, compliant consultation behavior.

## Purpose

The v4 reward design generally checks whether the assistant:

- understands the consultation context;
- transitions naturally toward contact collection;
- avoids unsafe or overly aggressive wording;
- follows the expected output format;
- satisfies rule-based constraints for contact-stage behavior.

## Typical Files

- contact-stage reward function scripts;
- rule-only variants for local debugging;
- reward-model wrapper scripts;
- GRPO launch scripts for Qwen3 8B experiments;
- test scripts for contact reward behavior.

## Inputs and Outputs

Inputs usually come from `project/rl_remake/` contact-stage dataset builders. Outputs are verl training directories under external paths such as `/data1/chengch/verl_outputs/` and should not be committed.

## When To Use

Use this folder for historical contact-stage reward experiments. For rubric-enhanced contact reward work, check `v5/`.
