# rl_remake

`rl_remake/` converts dialogue/SFT data into single-turn RL datasets that are easier for verl GRPO recipes to consume. Each script focuses on a different stage of the consultation dialogue.

## What This Folder Produces

The output rows usually contain fields such as:

- `prompt`: conversation context before the target assistant turn;
- `ground_truth`: reference assistant answer;
- `reward_model`: reward-related payload expected by verl;
- `extra_info`: stage, turn index, source, and auxiliary metadata;
- `data_source`, `agent_name`, `index`: compatibility fields for downstream training/evaluation.

Large generated train/val/test files are ignored by Git.

## Files

- `prepare_single_turn_rl_dataset_contact_stage.py`: original contact-stage dataset builder. Use it for earlier data sources and baseline contact experiments.
- `prepare_single_turn_rl_dataset_contact_stage_v2.py`: second contact-stage extraction flow with stricter stage handling.
- `prepare_single_turn_rl_dataset_contact_stage_new_sources.py`: contact-stage builder for newer source data.
- `prepare_single_turn_rl_dataset_contact_age_directed_v3.py`: directed contact-stage builder with age-related source/control handling.
- `prepare_single_turn_rl_dataset_mid_stage.py`: extracts mid-stage turns before explicit contact collection.
- `prepare_single_turn_rl_dataset_mid_stage_new_sources.py`: mid-stage builder for newer source data.
- `prepare_single_turn_rl_dataset_first2.py`: early-turn dataset construction, mainly for first/second-turn behavior experiments.
- `prepare_single_turn_rl_dataset_hard_sft_contact_mix_10k.py`: builds a hard-SFT/contact mixed dataset for supervised warmup or analysis.
- `stat_final_length.py`: statistics helper for final-response length distributions.

## Typical Workflow

1. Prepare or clean source conversations under `data_remake/`.
2. Pick the stage-specific builder in this folder.
3. Generate train/validation JSONL or parquet files.
4. Feed the outputs into `project/verl/recipe/single_turn_reward/` launch scripts.

Example:

```bash
cd /data/chengch/project
python rl_remake/prepare_single_turn_rl_dataset_contact_stage_new_sources.py
```

## Notes

Most scripts still use file-level constants or local default paths. Before running on a new machine, open the target script and check the input/output paths. Real API keys should never be placed in these files.
