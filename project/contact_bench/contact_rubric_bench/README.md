# Contact Rubric Bench

ITBench-style benchmark for single-turn contact-stage responses. It evaluates a candidate model response with contact-stage semantic rubrics plus deterministic hard penalties, then writes the same core artifacts as ITBench.

## Project Structure

```text
.
├── contactbench/          # local evaluation package
│   └── evaluation/        # rubric scorer
├── data/
│   ├── dataset/           # benchmark JSONL
│   ├── rubrics/           # semantic rubric set and index
│   └── rules/             # deterministic hard penalty config
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
data/dataset/single_turn_rl_contact_stage_new_sources_12k_age_directed.rubric_bench_300.jsonl
```

The evaluator accepts common prompt schemas:

- `prompt`, `messages`, or `raw_prompt` message arrays
- top-level `ground_truth` or `reward_model.ground_truth`
- existing candidate response fields such as `response`, `candidate_response`, `model_output`, `output`, or `solution_str`

## Run

Install dependencies in your evaluation environment:

```bash
pip install -r requirements.txt
```

Configure secrets outside Git:

```bash
cp env.example .env
```

Fill `JUDGE_API_KEY` and, for candidate generation, `CANDIDATE_*` values in `.env`, then run:

```bash
bash run_eval.sh
```

For a judge-only sanity run against ground-truth responses:

```bash
EVALUATION_LIMIT=1 bash run_eval.sh --use-ground-truth
```

CLI flags override environment values:

```bash
python scripts/evaluate_contact_rubric.py \
  --input-file data/dataset/single_turn_rl_contact_stage_new_sources_12k_age_directed.rubric_bench_300.jsonl \
  --rubric-path data/rubrics/contact_rubric_v001.json \
  --hard-config-path data/rules/contact_reward_hard_config.json \
  --output-root output \
  --limit 10 \
  --use-ground-truth
```

## Environment Contract

- `INPUT_FILE`
- `RUBRIC_PATH`
- `HARD_CONFIG_PATH`
- `OUTPUT`
- `EVALUATION_LIMIT`
- `CONCURRENCY`
- `CANDIDATE_MODEL_NAME`
- `CANDIDATE_API_BASE`
- `CANDIDATE_API_KEY`
- `JUDGE_MODEL_NAME`
- `JUDGE_API_BASE`
- `JUDGE_API_KEY`

## Outputs

Each run writes a timestamped directory under `output/`:

```text
<candidate>_<input_stem>_<rubric_stem>_<YYYYmmdd_HHMMSS>/
  config.json
  evaluation_results.jsonl
  failed_cases.jsonl
  excluded_generation_errors.jsonl
  evaluation_report.md
```

The report includes total samples, mean and median final score, mean semantic score, hard penalty statistics, generation status counts, judge status counts, and per-rubric trigger/pass/deduction summaries.
