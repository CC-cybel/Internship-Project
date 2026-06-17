# Internship-Project

This repository is a curated snapshot of the main internship workspaces used for dialogue data construction, single-turn RL training, reward design, and benchmark evaluation. It intentionally keeps source code, recipes, rules, rubrics, and documentation in Git, while excluding large generated datasets, logs, checkpoints, model weights, caches, and local secret files.

## Repository Map

```text
.
├── leadbench-excellent-master/   # dynamic sales/consultation dialogue benchmark
├── lm-evaluation-harness/        # local lm-eval fork plus leadbench evaluation extensions
└── project/                      # main RL/data/evaluation workspace
```

## Top-Level Folders

### `leadbench-excellent-master/`

Standalone benchmark for evaluating candidate models in optimization-style sales/consultation dialogues. It supports dynamic evaluation, where a user simulator talks to the candidate model and a judge model scores the full dialogue, and static evaluation, where completed dialogue histories are scored directly.

Important subfolders:

- `leadbench_excellent/`: core Python package.
- `leadbench_excellent/evaluation/`: rule scoring and session-level evaluation.
- `leadbench_excellent/generation/`: candidate response generation, model-specific preprocessing, and response postprocessing.
- `leadbench_excellent/model/`: API model wrapper.
- `leadbench_excellent/simulator/`: simulated user behavior and prompts.
- `leadbench_excellent/utils/`: config, dataset, and report helpers.
- `data/rules/`: benchmark rule JSON files.
- `data_prep/`: scripts and notes for creating benchmark inputs.
- `scripts/`: runnable entrypoints such as `dynamic_evaluate.py`.

Generated `output/` results are ignored by Git.

### `lm-evaluation-harness/`

Local copy of EleutherAI's Language Model Evaluation Harness with additional evaluation work under `evaluation/`. It provides standard LLM evaluation infrastructure and is also used as a home for local benchmark adapters.

Important subfolders:

- `lm_eval/`: upstream harness package and task/model/runtime implementation.
- `docs/`: upstream usage and configuration documentation.
- `evaluation/leadbench/`: local leadbench-style evaluation code and data preparation scripts.
- `evaluation/lm-evaluation-harness/`: local evaluation experiments based on the harness layout.
- `tests/`: upstream and local tests.

Use this folder when you need a reusable evaluation harness, task definitions, or leadbench-style evaluation logic outside the main RL workspace.

### `project/`

Main working directory for this internship project. It contains data cleaning and generation scripts, single-turn RL dataset builders, contact/instruction benchmarks, and verl recipes for GRPO/distillation experiments.

Important subfolders:

- `data_remake/`: dialogue data cleaning, rewriting, distillation, and SFT preparation.
- `rl_remake/`: conversion from dialogue/SFT data into single-turn RL datasets.
- `contact_bench/`: standalone benchmarks for contact-stage and instruction following behavior.
- `verl/`: local verl framework copy and custom training recipes.

The root script `run_benchmark_pipeline.sh` orchestrates several benchmark evaluations and report collection steps.

## `project/` Folder Details

### `project/data_remake/`

Data construction and cleaning workspace. This is where raw dialogue data is rewritten, filtered, converted, and packaged for SFT/RL experiments.

Common script groups:

- rewrite and reverse-prompt generation: `rewrite_dialogues.py`, `rewrite_dialogues1.py`, `rewrite_dialogues1_v2.py`, `reverse_prompts_from_dialogues.py`, `reverse_prompts_from_dialogues_v2.py`;
- Claude/OpenRouter-style distillation workflows: `claude_distill_contact_stage.py`, `claude_distill/claude_distill_contact_age_directed.py`, `run_claude_distill_contact_stage.sh`;
- quality checks and cleaning: `check_raw_hard_rewrite_quality.py`, `check_hard_rewrite_v2_object_quality.py`, `check_rewritten_hard_sft_quality.py`, `clean_rewritten_hard_sft_fixable.py`, `judge_reply_format_quality_with_model.py`;
- format conversion and preparation: `prepare_llamafactory_sft.py`, `convert_opd_jsonl_to_parquet.py`, `convert_meta_final_to_think_format.py`, `convert_to_dual_channel_format.py`, `convert_to_xml_channel_format.py`;
- instruction and slot-value data: `build_instruction_requirement_groups.py`, `chat_last_turn_value_slots_sft_v2.py`, `rewrite_last_turn_value_slots.py`.

Subfolders:

- `claude_distill/`: directed contact-age/contact-stage distillation scripts.
- `reference/`: reference material used by data-generation scripts.

Generated directories such as `outputs/`, `runs/`, `logs/`, `raw/`, `experiments/`, and large JSONL files are intentionally ignored.

### `project/rl_remake/`

Dataset builders for verl-compatible single-turn RL training. These scripts take multi-turn consultation dialogues and extract target assistant turns into RL samples with `prompt`, `ground_truth`, `reward_model`, and `extra_info` fields.

Key scripts:

- `prepare_single_turn_rl_dataset_contact_stage.py`: original contact-stage single-turn data builder.
- `prepare_single_turn_rl_dataset_contact_stage_v2.py`: updated contact-stage extraction flow.
- `prepare_single_turn_rl_dataset_contact_stage_new_sources.py`: contact-stage builder for newer data sources.
- `prepare_single_turn_rl_dataset_contact_age_directed_v3.py`: age-directed contact-stage dataset builder.
- `prepare_single_turn_rl_dataset_mid_stage.py` and `prepare_single_turn_rl_dataset_mid_stage_new_sources.py`: mid-stage samples before contact collection.
- `prepare_single_turn_rl_dataset_first2.py`: early-turn dataset construction.
- `prepare_single_turn_rl_dataset_hard_sft_contact_mix_10k.py`: mixed hard-SFT contact data preparation.
- `stat_final_length.py`: response length statistics.

Generated `rl_remake/outputs/` files are ignored because they are usually large.

### `project/contact_bench/`

Standalone benchmark workspace for evaluating generated responses without running full RL training.

Subfolders:

- `contact_rubric_bench/`: ITBench-style contact-stage rubric benchmark. It combines semantic rubric scoring with deterministic hard penalties and writes `evaluation_results.jsonl`, `failed_cases.jsonl`, `excluded_generation_errors.jsonl`, `evaluation_report.md`, and `config.json`.
- `instruction_following_bench/`: benchmark for checking whether a candidate response follows injected system instructions. It can generate candidate responses online, judge existing outputs offline, or score reference responses for sanity checks.

Both benchmarks use `env.example` for environment variables. Real API keys should be supplied through local `.env` files or shell variables, not committed to Git.

### `project/verl/`

Local copy of the verl training framework with custom recipes for this project. Most project-specific work lives under `project/verl/recipe/`.

Important recipe folders:

- `recipe/single_turn_reward/`: GRPO training recipes and reward functions for single-turn consultation behavior.
- `recipe/opd_multi_teacher/`: multi-teacher on-policy distillation experiments.
- `recipe/dual_zscore_grpo/`: experimental dual-z-score advantage logic and tests.

Framework folders such as `verl/`, `examples/`, `docs/`, `tests/`, and `docker/` come from the underlying verl project and support training, rollout, workers, configuration, and model utilities.

### `project/verl/recipe/single_turn_reward/`

Versioned reward/training recipes for staged single-turn RL:

- `origin/`: early baseline reward/template code.
- `v1/` to `v5/`: staged versions of reward functions, reward-model wrappers, evaluation utilities, and launch scripts.
- `config/`: recipe-level config assets.
- `rubrics_instruction_following.json`, `talk_eval_rule.json`, `bench_excellent.json`: rule/rubric assets used by reward or evaluation logic.

Recent active pieces include contact/rubric reward training scripts in `v5/`, contact-stage and mid-stage launch scripts in earlier version folders, and offline scoring/rubric-review utilities in `v5/`.

Training outputs, logs, checkpoint folders, and collected data are ignored.

### `project/verl/recipe/opd_multi_teacher/`

Multi-teacher on-policy distillation recipe. It routes samples to different teacher models based on a `teacher_route` field and optimizes student behavior with KL-style distillation losses.

Main files:

- `build_opd_multi_teacher_data.py`: builds mixed contact/mid-stage training data.
- `reward_zero.py`: zero task-reward function for pure distillation runs.
- `run_opd_multi_teacher_qwen3_8b.sh`: launch script for the original recipe.
- `v2/`: forward-KL/top-k teacher-forced variant.
- `v3/`: reverse-KL/top-k variant.

Data files under `data/`, `v2/data/`, and `v3/data/` are ignored.

### `project/verl/recipe/dual_zscore_grpo/`

Experimental GRPO advantage normalization recipe. It contains:

- `dual_zscore_advantage.py`: dual-z-score advantage computation.
- `main_ppo.py`: training entrypoint variant.
- `run_grpo_single_turn_4gpu_qwen3_8b_stage4_dual_zscore.sh`: launch script.
- `tests/`: toy/random/JSONL alpha sweep tests and unit checks.

Use this folder when comparing standard GRPO advantage computation with dual-z-score variants.

## Git and Data Hygiene

The repository intentionally excludes:

- Python caches and build products;
- `.env`, credentials, certificates, and secret files;
- generated outputs, logs, benchmark results, SwanLab/W&B/TensorBoard files;
- checkpoint/model artifacts such as `.safetensors`, `.pt`, `.bin`, `.ckpt`;
- large JSONL/Parquet/Arrow datasets and PDFs.

If a script needs an API key, pass it through an environment variable or an untracked `.env` file. Do not add real keys as default values in scripts.

## Common Commands

Check status:

```bash
cd /data/chengch/Internship-Project
git status
```

Push to GitHub:

```bash
git push origin main
```

Push the same snapshot to GitLab:

```bash
git remote set-url gitlab http://vpn.shengwenyun.cn/chengchong/project.git
git push -u gitlab main:master
```

If the GitLab default branch is `main`, use:

```bash
git push -u gitlab main:main
```

## Notes for Future Work

- Keep generated data in ignored local folders or external storage.
- Keep benchmark input samples small enough for Git, and place large full datasets outside this repository.
- When adding a new experiment folder, include a short local README or comments explaining the entry script, expected inputs, and output path.
- Before pushing, scan for accidental secrets and large files.
