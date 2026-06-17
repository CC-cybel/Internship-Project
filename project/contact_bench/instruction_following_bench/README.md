# Instruction Following Bench

ITBench-style benchmark for evaluating whether a candidate model can follow injected system instructions. The default path calls the candidate model to generate a response, then calls a judge model to score instruction following. Each sample contains a normal consultation dialogue plus `selected_additional_instructions`. The evaluator asks a judge model to score every injected instruction on a 0/1/2 scale, then reports sample-level, bucket-level, atom-level, and failure-type summaries.

## Project Structure

```text
.
├── instructionbench/      # reserved local package namespace
│   └── evaluation/
├── data/
│   └── dataset/           # default 300-sample instruction-following bench
├── output/                # timestamped evaluation outputs
├── scripts/               # evaluation entrypoints
├── env.example            # environment template, no real secrets
├── run_eval.sh            # shell wrapper
├── requirements.txt
└── README.md
```

## Inputs

Default dataset:

```text
data/dataset/instruction_following_bench_final300.jsonl
```

Expected core fields per row:

- `system`: full system prompt with appended instruction block
- `conversations`: dialogue history ending at the target assistant turn
- `selected_additional_instructions`: injected instruction objects to judge
- `selected_instruction_ids`: selected atom ids
- `target.original_value`: reference/original response, used only when `--use-reference` or `USE_REFERENCE=1`

For real model evaluation, either add a response field such as `model_output`, `response`, `candidate_response`, `output`, or `solution_str`, then set `RESPONSE_FIELD` / `--response-field`; or configure `CANDIDATE_*` so the runner generates candidate responses online before judging.

## Scoring

Each injected instruction is scored independently:

- `0`: not followed, contradicted, or impossible to verify because required content is missing
- `1`: partially followed, weakly followed, or minor but meaningful issue
- `2`: fully followed, or a negative/prohibition constraint is not violated

The sample score is a weighted mean of `score / 2`. The evaluator also reports an unweighted score, strict pass rate, soft pass rate, format pass rate, and breakdowns by instruction count, source, axis, atom id, and failure type.

## Run

Configure secrets outside Git:

```bash
cp env.example .env
```

Fill `JUDGE_API_KEY`, then run a judge-only sanity check against the reference/original responses:

```bash
USE_REFERENCE=1 EVALUATION_LIMIT=10 bash run_eval.sh
```

Default: generate candidate responses online, then judge them:

```bash
CANDIDATE_MODEL_NAME=qwen3_8b_rubric_rl_step540 \
CANDIDATE_API_BASE=http://127.0.0.1:8002/v1 \
CANDIDATE_API_KEY=EMPTY \
RUN_NAME=qwen3_8b_instruction_following \
bash run_eval.sh
```

Offline mode for already-generated model outputs:

```bash
INPUT_FILE=/path/to/model_outputs.jsonl \
RESPONSE_FIELD=model_output \
RUN_NAME=my_model_offline_instruction_following \
bash run_eval.sh
```

CLI flags can also be passed through:

```bash
bash run_eval.sh --use-reference --limit 5 --run-name probe5_reference
```

## Environment Contract

- `INPUT_FILE`
- `OUTPUT`
- `EVALUATION_LIMIT`
- `OFFSET`
- `CONCURRENCY`
- `EVAL_RETRIES`
- `USE_REFERENCE`
- `RESPONSE_FIELD`
- `RUN_NAME`
- `CANDIDATE_MODEL_NAME`
- `CANDIDATE_API_BASE`
- `CANDIDATE_API_KEY`
- `CANDIDATE_MAX_OUTPUT_TOKENS`
- `CANDIDATE_TEMPERATURE`
- `CANDIDATE_TOP_P`
- `CANDIDATE_TIMEOUT_S`
- `PREFLIGHT_CANDIDATE`
- `JUDGE_MODEL_NAME`
- `JUDGE_API_BASE`
- `JUDGE_API_KEY`
- `JUDGE_MAX_TOKENS`
- `JUDGE_TIMEOUT_S`
- `JUDGE_TEMPERATURE`

## Outputs

Each run writes a timestamped directory under `output/`, or a named directory when `RUN_NAME`/`--run-name` is set:

```text
<run_name>/
  config.json
  evaluation_results.jsonl
  failed_cases.jsonl
  excluded_generation_errors.jsonl
  evaluation_report.md
  summary.json
```

`evaluation_results.jsonl` contains one row per sample with generated candidate responses when applicable, candidate usage, and per-instruction judge details. `failed_cases.jsonl` contains non-OK rows and rows whose sample score is below 1.0. `excluded_generation_errors.jsonl` contains rows missing candidate responses.

The report includes total samples, mean/median score, strict and soft pass rates, format pass rate, score by 1/2/3 instruction buckets, score by source/axis, failure types, and lowest atom scores.
