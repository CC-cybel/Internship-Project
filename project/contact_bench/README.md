# contact_bench

`contact_bench/` contains standalone benchmarks for evaluating consultation model behavior without launching a full RL training job. The goal is to provide small, repeatable, report-producing evaluations that can be run against a deployed candidate model or against already-generated outputs.

## Subfolders

### `contact_rubric_bench/`

ITBench-style benchmark for contact-stage responses. It evaluates whether a response handles contact collection naturally and safely.

Main responsibilities:

- load a single-turn contact-stage benchmark dataset;
- optionally call a candidate model to generate responses;
- call a judge model to score semantic rubrics;
- apply deterministic hard penalties from rule config;
- write `evaluation_results.jsonl`, `failed_cases.jsonl`, `excluded_generation_errors.jsonl`, `evaluation_report.md`, and `config.json`.

Key files:

- `run_eval.sh`: shell wrapper for normal runs.
- `scripts/evaluate_contact_rubric.py`: main evaluator.
- `contactbench/evaluation/rubric_scorer.py`: rubric scoring logic.
- `data/rubrics/contact_rubric_v001.json`: semantic rubric definition.
- `data/rules/contact_reward_hard_config.json`: deterministic penalty config.
- `env.example`: environment variable template; do not commit real secrets.

### `instruction_following_bench/`

Benchmark for instruction-following behavior when additional system instructions are injected into a consultation prompt.

Main responsibilities:

- load samples with `selected_additional_instructions`;
- generate candidate responses online or read an existing response field;
- judge every injected instruction on a 0/1/2 scale;
- summarize scores by instruction count, source, axis, atom id, and failure type;
- write machine-readable outputs and a markdown report.

Key files:

- `run_eval.sh`: shell wrapper.
- `scripts/evaluate_instruction_following.py`: main evaluator.
- `scripts/compare_instruction_following_runs.py`: compare two or more runs.
- `scripts/visualize_instruction_following.py`: chart/report helper.
- `BUILD_AND_USAGE.md`: detailed build and usage notes.
- `env.example`: environment variable template.

## Typical Usage

Run a small reference sanity check:

```bash
cd /data/chengch/project/contact_bench/instruction_following_bench
USE_REFERENCE=1 EVALUATION_LIMIT=10 bash run_eval.sh
```

Run contact rubric evaluation:

```bash
cd /data/chengch/project/contact_bench/contact_rubric_bench
EVALUATION_LIMIT=10 bash run_eval.sh --use-ground-truth
```

## Git Policy

Generated `output/` directories, large JSONL result files, and local `.env` files are ignored. Keep only benchmark code, small configs, rubrics, rules, and documentation in Git.
