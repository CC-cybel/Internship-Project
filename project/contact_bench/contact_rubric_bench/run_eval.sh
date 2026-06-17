#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0,::1}"
export no_proxy="${NO_PROXY}"

export INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/data/dataset/single_turn_rl_contact_stage_new_sources_12k_age_directed.rubric_bench_300.jsonl}"
export RUBRIC_PATH="${RUBRIC_PATH:-${ROOT_DIR}/data/rubrics/contact_rubric_v001.json}"
export HARD_CONFIG_PATH="${HARD_CONFIG_PATH:-${ROOT_DIR}/data/rules/contact_reward_hard_config.json}"
export OUTPUT="${OUTPUT:-${ROOT_DIR}/output}"

export EVALUATION_LIMIT="${EVALUATION_LIMIT:-0}"
export CONCURRENCY="${CONCURRENCY:-20}"
export EVAL_RETRIES="${EVAL_RETRIES:-4}"
export PREFLIGHT_CANDIDATE="${PREFLIGHT_CANDIDATE:-1}"
export FAIL_ON_GENERATION_ERROR="${FAIL_ON_GENERATION_ERROR:-1}"

export CANDIDATE_MODEL_NAME="${CANDIDATE_MODEL_NAME:-}"
export CANDIDATE_API_BASE="${CANDIDATE_API_BASE:-}"
export CANDIDATE_API_KEY="${CANDIDATE_API_KEY:-}"
export CANDIDATE_MAX_OUTPUT_TOKENS="${CANDIDATE_MAX_OUTPUT_TOKENS:-2048}"
export CANDIDATE_TEMPERATURE="${CANDIDATE_TEMPERATURE:-0.6}"
export CANDIDATE_TOP_P="${CANDIDATE_TOP_P:-0.95}"
export CANDIDATE_TIMEOUT_S="${CANDIDATE_TIMEOUT_S:-180}"

export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1600}"
export JUDGE_TIMEOUT_S="${JUDGE_TIMEOUT_S:-45}"

DEFAULT_PYTHON="/data/wangpf/project/miniconda3/envs/lm-evaluation-harness/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${EVAL_PYTHON:-${PYTHON:-${DEFAULT_PYTHON}}}"
else
  PYTHON="${EVAL_PYTHON:-${PYTHON:-python3}}"
fi
SCRIPT="${ROOT_DIR}/scripts/evaluate_contact_rubric.py"

echo "Starting contact rubric bench..."
echo "Candidate model: ${CANDIDATE_MODEL_NAME:-<unset>}"
echo "Candidate API: ${CANDIDATE_API_BASE:-<unset>}"
echo "Judge model: ${JUDGE_MODEL_NAME:-<unset>}"
echo "Judge API: ${JUDGE_API_BASE:-<unset>}"
echo "Input file: ${INPUT_FILE}"
echo "Rubric: ${RUBRIC_PATH}"
echo "Hard config: ${HARD_CONFIG_PATH}"
echo "Output root: ${OUTPUT}"
echo "Evaluation limit: ${EVALUATION_LIMIT}"
echo "Concurrency: ${CONCURRENCY}"

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

if [[ -z "${JUDGE_MODEL_NAME}" || -z "${JUDGE_API_BASE}" || -z "${JUDGE_API_KEY}" ]]; then
  echo "ERROR: set JUDGE_MODEL_NAME, JUDGE_API_BASE, and JUDGE_API_KEY before running evaluation." >&2
  echo "Tip: copy env.example to .env and fill the secrets, or export them in your shell/CI." >&2
  exit 1
fi

if [[ "${PREFLIGHT_CANDIDATE}" == "1" && "${USE_GROUND_TRUTH}" == "0" ]]; then
  if [[ -z "${CANDIDATE_MODEL_NAME}" || -z "${CANDIDATE_API_BASE}" || -z "${CANDIDATE_API_KEY}" ]]; then
    echo "ERROR: candidate generation needs CANDIDATE_MODEL_NAME, CANDIDATE_API_BASE, and CANDIDATE_API_KEY." >&2
    echo "Tip: use --use-ground-truth for judge-only smoke tests." >&2
    exit 1
  fi
  CANDIDATE_MODELS_URL="${CANDIDATE_API_BASE%/}"
  if [[ "${CANDIDATE_MODELS_URL}" == */v1 ]]; then
    CANDIDATE_MODELS_URL="${CANDIDATE_MODELS_URL}/models"
  else
    CANDIDATE_MODELS_URL="${CANDIDATE_MODELS_URL}/v1/models"
  fi
  echo "Preflight candidate API: ${CANDIDATE_MODELS_URL}"
  if ! curl --connect-timeout 5 --max-time 15 -fsS -H "Authorization: Bearer ${CANDIDATE_API_KEY}" "${CANDIDATE_MODELS_URL}" >/dev/null; then
    echo "ERROR: candidate API preflight failed: ${CANDIDATE_MODELS_URL}" >&2
    exit 2
  fi
fi

"${PYTHON}" "${SCRIPT}" "$@"

echo "Evaluation finished."
