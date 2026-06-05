#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0,::1}"
export no_proxy="${NO_PROXY}"

export CANDIDATE_MODEL_NAME="${CANDIDATE_MODEL_NAME:-qwen3_8b_rubric_rl_step540}"
export CANDIDATE_API_BASE="${CANDIDATE_API_BASE:-http://127.0.0.1:8002/v1}"
export CANDIDATE_API_KEY="${CANDIDATE_API_KEY:-token-eval123}"

export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-deepseek-v4-flash}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"

export INPUT_FILE="${INPUT_FILE:-/data/chengch/project/verl/recipe/single_turn_reward/v5/data/single_turn_rl_contact_stage_new_sources_12k_age_directed.rubric_bench_300.jsonl}"
export RUBRIC_PATH="${RUBRIC_PATH:-/data/chengch/project/verl/recipe/single_turn_reward/v5/rubrics/contact_rubric_v001.json}"
export HARD_CONFIG_PATH="${HARD_CONFIG_PATH:-/data/chengch/project/verl/recipe/single_turn_reward/v5/contact_reward_hard_config.json}"
export OUTPUT="${OUTPUT:-/data/chengch/project/verl/recipe/single_turn_reward/v5/data/rubric_eval_outputs}"

export EVALUATION_LIMIT="${EVALUATION_LIMIT:-0}"  # 0 means all samples.
export CONCURRENCY="${CONCURRENCY:-20}"
export EVAL_RETRIES="${EVAL_RETRIES:-4}"
export PREFLIGHT_CANDIDATE="${PREFLIGHT_CANDIDATE:-1}"
export FAIL_ON_GENERATION_ERROR="${FAIL_ON_GENERATION_ERROR:-1}"

export CANDIDATE_MAX_OUTPUT_TOKENS="${CANDIDATE_MAX_OUTPUT_TOKENS:-2048}"
export CANDIDATE_TEMPERATURE="${CANDIDATE_TEMPERATURE:-0.6}"
export CANDIDATE_TOP_P="${CANDIDATE_TOP_P:-0.95}"
export CANDIDATE_TIMEOUT_S="${CANDIDATE_TIMEOUT_S:-180}"

export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1600}"
export JUDGE_TIMEOUT_S="${JUDGE_TIMEOUT_S:-45}"

DEFAULT_PYTHON="/data/wangpf/project/miniconda3/envs/lm-evaluation-harness/bin/python"
PYTHON="${EVAL_PYTHON:-${PYTHON:-${DEFAULT_PYTHON}}}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON}" >&2
  echo "ERROR: expected lm-evaluation-harness at ${DEFAULT_PYTHON}; set EVAL_PYTHON=/path/to/python and retry." >&2
  exit 1
fi
SCRIPT="/data/chengch/project/verl/recipe/single_turn_reward/v5/evaluate_contact_rubric_bench.py"

echo "开始 contact rubric bench 评测..."
echo "Candidate model: ${CANDIDATE_MODEL_NAME}"
echo "Candidate API: ${CANDIDATE_API_BASE}"
echo "Judge model: ${JUDGE_MODEL_NAME}"
echo "Judge API: ${JUDGE_API_BASE}"
echo "Input file: ${INPUT_FILE}"
echo "Rubric: ${RUBRIC_PATH}"
echo "Output root: ${OUTPUT}"
echo "Evaluation limit: ${EVALUATION_LIMIT}"
echo "Concurrency: ${CONCURRENCY}"
echo "Retries: ${EVAL_RETRIES}"
echo "Candidate max tokens: ${CANDIDATE_MAX_OUTPUT_TOKENS}"
echo "Judge max tokens: ${JUDGE_MAX_TOKENS}"
echo "Fail on generation error: ${FAIL_ON_GENERATION_ERROR}"
echo "Candidate preflight: ${PREFLIGHT_CANDIDATE}"

USE_GROUND_TRUTH=0
SHOW_HELP=0
for arg in "$@"; do
  if [[ "${arg}" == "--use-ground-truth" ]]; then
    USE_GROUND_TRUTH=1
  fi
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    SHOW_HELP=1
  fi
done

if [[ "${SHOW_HELP}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT}" "$@"
  exit 0
fi

if [[ "${PREFLIGHT_CANDIDATE}" == "1" && "${USE_GROUND_TRUTH}" == "0" ]]; then
  CANDIDATE_MODELS_URL="${CANDIDATE_API_BASE%/}"
  if [[ "${CANDIDATE_MODELS_URL}" == */v1 ]]; then
    CANDIDATE_MODELS_URL="${CANDIDATE_MODELS_URL}/models"
  else
    CANDIDATE_MODELS_URL="${CANDIDATE_MODELS_URL}/v1/models"
  fi
  echo "Preflight candidate API: ${CANDIDATE_MODELS_URL}"
  if ! curl --connect-timeout 5 --max-time 15 -fsS -H "Authorization: Bearer ${CANDIDATE_API_KEY}" "${CANDIDATE_MODELS_URL}" >/dev/null; then
    echo "ERROR: candidate API preflight failed: ${CANDIDATE_MODELS_URL}" >&2
    echo "ERROR: 请确认 vLLM 服务还在运行，且 CANDIDATE_API_BASE/端口配置正确。" >&2
    exit 2
  fi
fi

"${PYTHON}" "${SCRIPT}" "$@"

echo "评测完成。"
