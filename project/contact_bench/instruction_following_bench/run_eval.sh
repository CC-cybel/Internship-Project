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

export INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/data/dataset/instruction_following_bench_final300.jsonl}"
export OUTPUT="${OUTPUT:-${ROOT_DIR}/output}"
export EVALUATION_LIMIT="${EVALUATION_LIMIT:-0}"
export OFFSET="${OFFSET:-0}"
export CONCURRENCY="${CONCURRENCY:-4}"
export EVAL_RETRIES="${EVAL_RETRIES:-4}"

export RESPONSE_FIELD="${RESPONSE_FIELD:-}"
export USE_REFERENCE="${USE_REFERENCE:-0}"
export RUN_NAME="${RUN_NAME:-}"

export CANDIDATE_MODEL_NAME="${CANDIDATE_MODEL_NAME:-qwen3_8b_sft_v2}"
export CANDIDATE_API_BASE="${CANDIDATE_API_BASE:-http://127.0.0.1:8001/v1}"
export CANDIDATE_API_KEY="${CANDIDATE_API_KEY:-111}"
export CANDIDATE_MAX_OUTPUT_TOKENS="${CANDIDATE_MAX_OUTPUT_TOKENS:-768}"
export CANDIDATE_TEMPERATURE="${CANDIDATE_TEMPERATURE:-0.6}"
export CANDIDATE_TOP_P="${CANDIDATE_TOP_P:-0.95}"
export CANDIDATE_TIMEOUT_S="${CANDIDATE_TIMEOUT_S:-180}"
export PREFLIGHT_CANDIDATE="${PREFLIGHT_CANDIDATE:-1}"

export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-deepseek-v4-flash}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-your_api_key_here}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_TIMEOUT_S="${JUDGE_TIMEOUT_S:-120}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"

# Convenience fallback for this workspace: reuse the same non-committed API env source
# used by the instruction generation scripts when .env has not been filled yet.
FALLBACK_ENV_FILE="${FALLBACK_ENV_FILE:-/data/chengch/project/verl/recipe/single_turn_reward/v3/run_grpo_single_turn_4gpu_qwen3_8b_stage5_mid_reward.sh}"
if [[ -z "${JUDGE_API_KEY}" && -f "${FALLBACK_ENV_FILE}" ]]; then
  FALLBACK_ASSIGNMENTS="$(python3 - "${FALLBACK_ENV_FILE}" <<'PYFALLBACK'
import re
import shlex
import sys
from pathlib import Path
path = Path(sys.argv[1])
parsed = {}
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    m = re.match(r':\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=([^}]*)\}"', line)
    if m:
        parsed[m.group(1)] = m.group(2)
for out_key, in_key in [("JUDGE_API_KEY", "TONGYI_API_KEY"), ("JUDGE_API_BASE", "TONGYI_API_BASE"), ("JUDGE_MODEL_NAME", "JUDGE_MODEL"), ("CANDIDATE_API_KEY", "TONGYI_API_KEY"), ("CANDIDATE_API_BASE", "TONGYI_API_BASE")]:
    if parsed.get(in_key):
        print(f"export {out_key}={shlex.quote(parsed[in_key])}")
PYFALLBACK
)"
  if [[ -n "${FALLBACK_ASSIGNMENTS}" ]]; then
    eval "${FALLBACK_ASSIGNMENTS}"
  fi
fi

DEFAULT_PYTHON="/data/chengch/.conda/envs/verl/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${EVAL_PYTHON:-${PYTHON:-${DEFAULT_PYTHON}}}"
else
  PYTHON="${EVAL_PYTHON:-${PYTHON:-python3}}"
fi
SCRIPT="${ROOT_DIR}/scripts/evaluate_instruction_following.py"

SHOW_HELP=0
USE_REFERENCE_ARG=0
for arg in "$@"; do
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    SHOW_HELP=1
  fi
  if [[ "${arg}" == "--use-reference" ]]; then
    USE_REFERENCE_ARG=1
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

ARGS=(
  --input-file "${INPUT_FILE}"
  --output-root "${OUTPUT}"
  --api-base "${JUDGE_API_BASE}"
  --api-key "${JUDGE_API_KEY}"
  --judge-model "${JUDGE_MODEL_NAME}"
  --candidate-model "${CANDIDATE_MODEL_NAME}"
  --candidate-api-base "${CANDIDATE_API_BASE}"
  --candidate-api-key "${CANDIDATE_API_KEY}"
  --candidate-max-tokens "${CANDIDATE_MAX_OUTPUT_TOKENS}"
  --candidate-temperature "${CANDIDATE_TEMPERATURE}"
  --candidate-top-p "${CANDIDATE_TOP_P}"
  --candidate-timeout "${CANDIDATE_TIMEOUT_S}"
  --concurrency "${CONCURRENCY}"
  --max-retries "${EVAL_RETRIES}"
  --timeout "${JUDGE_TIMEOUT_S}"
  --temperature "${JUDGE_TEMPERATURE}"
  --max-tokens "${JUDGE_MAX_TOKENS}"
  --offset "${OFFSET}"
)

if [[ "${EVALUATION_LIMIT}" != "0" && -n "${EVALUATION_LIMIT}" ]]; then
  ARGS+=(--limit "${EVALUATION_LIMIT}")
fi
if [[ -n "${RESPONSE_FIELD}" ]]; then
  ARGS+=(--response-field "${RESPONSE_FIELD}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  ARGS+=(--run-name "${RUN_NAME}")
fi
if [[ "${USE_REFERENCE}" == "1" || "${USE_REFERENCE_ARG}" == "1" ]]; then
  ARGS+=(--use-reference)
fi

if [[ "${USE_REFERENCE}" != "1" && "${USE_REFERENCE_ARG}" != "1" && -z "${RESPONSE_FIELD}" ]]; then
  if [[ -z "${CANDIDATE_MODEL_NAME}" || -z "${CANDIDATE_API_BASE}" || -z "${CANDIDATE_API_KEY}" ]]; then
    echo "ERROR: set RESPONSE_FIELD for existing model outputs, or set CANDIDATE_MODEL_NAME/CANDIDATE_API_BASE/CANDIDATE_API_KEY for candidate generation, or run USE_REFERENCE=1 for sanity checks." >&2
    exit 1
  fi
fi

if [[ "${PREFLIGHT_CANDIDATE}" == "1" && "${USE_REFERENCE}" != "1" && "${USE_REFERENCE_ARG}" != "1" && -z "${RESPONSE_FIELD}" ]]; then
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

echo "Starting instruction-following bench..."
echo "Candidate model: ${CANDIDATE_MODEL_NAME:-<unset>}"
echo "Candidate API: ${CANDIDATE_API_BASE:-<unset>}"
echo "Judge model: ${JUDGE_MODEL_NAME}"
echo "Judge API: ${JUDGE_API_BASE}"
echo "Input file: ${INPUT_FILE}"
echo "Output root: ${OUTPUT}"
echo "Evaluation limit: ${EVALUATION_LIMIT}"
echo "Concurrency: ${CONCURRENCY}"
echo "Response field: ${RESPONSE_FIELD:-<unset>}"
echo "Use reference: ${USE_REFERENCE}"

"${PYTHON}" "${SCRIPT}" "${ARGS[@]}" "$@"

echo "Evaluation finished."
