#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="${PIPELINE_ROOT:-/data/chengch/project/benchmark_pipeline_outputs}"
TIMESTAMP="${PIPELINE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${PIPELINE_ROOT}/${TIMESTAMP}}"
REPORT_DIR="${RUN_DIR}/reports"
LOG_DIR="${RUN_DIR}/logs"

START_VLLM_SCRIPT="${START_VLLM_SCRIPT:-/data/chengch/lm-evaluation-harness/itbench-master/itbench-master/start_vllm.sh}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:8004/v1/models}"

CONTACT_SCRIPT="${CONTACT_SCRIPT:-/data/chengch/project/verl/recipe/single_turn_reward/v5/run_evaluate_contact_rubric_bench.sh}"
ITBENCH_ROOT="${ITBENCH_ROOT:-/data/chengch/lm-evaluation-harness/itbench-master/itbench-master}"
ITBENCH_SCRIPT="${ITBENCH_SCRIPT:-${ITBENCH_ROOT}/run_eval.sh}"
LEADBENCH_ROOT="${LEADBENCH_ROOT:-/data/chengch/lm-evaluation-harness/evaluation/leadbench}"
LEADBENCH_SCRIPT="${LEADBENCH_SCRIPT:-${LEADBENCH_ROOT}/scripts/evaluate_golden_history.py}"
LEADBENCH_EXCELLENT_ROOT="${LEADBENCH_EXCELLENT_ROOT:-/data/chengch/leadbench-excellent-master}"
LEADBENCH_EXCELLENT_PACKAGE="${LEADBENCH_EXCELLENT_PACKAGE:-${LEADBENCH_EXCELLENT_ROOT}/leadbench_excellent}"
LEADBENCH_EXCELLENT_SCRIPT="${LEADBENCH_EXCELLENT_SCRIPT:-${LEADBENCH_EXCELLENT_ROOT}/scripts/dynamic_evaluate.py}"
EVAL_PYTHON="${EVAL_PYTHON:-/data/wangpf/project/miniconda3/envs/lm-evaluation-harness/bin/python}"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0,::1}"
export no_proxy="${NO_PROXY}"

export CANDIDATE_MODEL_NAME="${CANDIDATE_MODEL_NAME:-qwen3_8b_rubric_rl_step540}"
export CANDIDATE_API_BASE="${CANDIDATE_API_BASE:-http://127.0.0.1:8004/v1}"
export CANDIDATE_API_KEY="${CANDIDATE_API_KEY:-token-eval123}"
export CANDIDATE_MAX_OUTPUT_TOKENS="${CANDIDATE_MAX_OUTPUT_TOKENS:-512}"

export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-deepseek-v4-flash}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-your_api_key_here}"
export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-false}"
export USER_SIMULATOR_MODEL_NAME="${USER_SIMULATOR_MODEL_NAME:-${JUDGE_MODEL_NAME}}"
export USER_SIMULATOR_API_BASE="${USER_SIMULATOR_API_BASE:-${JUDGE_API_BASE}}"
export USER_SIMULATOR_API_KEY="${USER_SIMULATOR_API_KEY:-${JUDGE_API_KEY}}"

export CONCURRENCY="${CONCURRENCY:-20}"
export EVALUATION_LIMIT="${EVALUATION_LIMIT:-}"

CONTACT_OUTPUT="${CONTACT_OUTPUT:-${RUN_DIR}/contact_rubric}"
ITBENCH_OUTPUT="${ITBENCH_OUTPUT:-${RUN_DIR}/itbench}"
LEADBENCH_OUTPUT="${LEADBENCH_OUTPUT:-${RUN_DIR}/leadbench}"
LEADBENCH_EXCELLENT_OUTPUT="${LEADBENCH_EXCELLENT_OUTPUT:-${RUN_DIR}/leadbench_excellent}"

ITBENCH_INPUT_FILE="${ITBENCH_INPUT_FILE:-./data/dataset/psychiatry/golden_history_input.jsonl}"
ITBENCH_RULES_FILE="${ITBENCH_RULES_FILE:-./data/rules/itbench_rule.json}"
LEADBENCH_INPUT_FILE="${LEADBENCH_INPUT_FILE:-./data/dataset/psychiatry/golden_history_input_v1.jsonl}"
LEADBENCH_RULES_FILE="${LEADBENCH_RULES_FILE:-./data/rules/leadbench_rule.json}"
LEADBENCH_EXCELLENT_INPUT_FILE="${LEADBENCH_EXCELLENT_INPUT_FILE:-./data/dataset/search_word_input_v1.jsonl}"
LEADBENCH_EXCELLENT_RULES_FILE="${LEADBENCH_EXCELLENT_RULES_FILE:-./data/rules/leadbench_excellent_rule.json}"

RUN_CONTACT="${RUN_CONTACT:-1}"
RUN_ITBENCH="${RUN_ITBENCH:-1}"
RUN_LEADBENCH="${RUN_LEADBENCH:-1}"
RUN_LEADBENCH_EXCELLENT="${RUN_LEADBENCH_EXCELLENT:-1}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-0}"

usage() {
  cat <<'USAGE'
Usage:
  bash /data/chengch/project/run_benchmark_pipeline.sh [--skip-vllm] [--contact-only|--itbench-only|--leadbench-only|--leadbench-excellent-only]

Useful env overrides:
  PIPELINE_ROOT=/path/to/outputs
  RUN_DIR=/path/to/specific/run
  CANDIDATE_MODEL_NAME=...
  CANDIDATE_API_BASE=http://127.0.0.1:8004/v1
  CANDIDATE_API_KEY=...
  JUDGE_MODEL_NAME=...
  JUDGE_API_BASE=...
  JUDGE_API_KEY=...
  USER_SIMULATOR_MODEL_NAME=...
  USER_SIMULATOR_API_BASE=...
  USER_SIMULATOR_API_KEY=...
  CONCURRENCY=20
  EVALUATION_LIMIT=10
  STOP_VLLM_ON_EXIT=1

Output layout:
  $RUN_DIR/contact_rubric/<bench-run>/
  $RUN_DIR/itbench/<bench-run>/
  $RUN_DIR/leadbench/<bench-run>/
  $RUN_DIR/leadbench_excellent/<bench-run>/
  $RUN_DIR/reports/
  $RUN_DIR/logs/
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-vllm)
      START_VLLM=0
      shift
      ;;
    --contact-only)
      RUN_CONTACT=1
      RUN_ITBENCH=0
      RUN_LEADBENCH=0
      RUN_LEADBENCH_EXCELLENT=0
      shift
      ;;
    --itbench-only)
      RUN_CONTACT=0
      RUN_ITBENCH=1
      RUN_LEADBENCH=0
      RUN_LEADBENCH_EXCELLENT=0
      shift
      ;;
    --leadbench-only)
      RUN_CONTACT=0
      RUN_ITBENCH=0
      RUN_LEADBENCH=1
      RUN_LEADBENCH_EXCELLENT=0
      shift
      ;;
    --leadbench-excellent-only|--leadbench_excel-only|--leadbench-excellent)
      RUN_CONTACT=0
      RUN_ITBENCH=0
      RUN_LEADBENCH=0
      RUN_LEADBENCH_EXCELLENT=1
      shift
      ;;
    --skip-leadbench-excellent)
      RUN_LEADBENCH_EXCELLENT=0
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${CONTACT_OUTPUT}" "${ITBENCH_OUTPUT}" "${LEADBENCH_OUTPUT}" "${LEADBENCH_EXCELLENT_OUTPUT}"

VLLM_PID=""
cleanup() {
  if [[ "${STOP_VLLM_ON_EXIT}" == "1" && -n "${VLLM_PID}" ]]; then
    echo "[pipeline] stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

log_step() {
  echo
  echo "========== $* =========="
}

health_check() {
  curl --connect-timeout 5 --max-time 15 -fsS "${VLLM_HEALTH_URL}" >/dev/null
}

ensure_vllm() {
  if health_check; then
    echo "[pipeline] vLLM already ready: ${VLLM_HEALTH_URL}"
    return 0
  fi
  if [[ "${START_VLLM}" != "1" ]]; then
    echo "ERROR: vLLM is not ready and START_VLLM=0: ${VLLM_HEALTH_URL}" >&2
    exit 3
  fi

  log_step "Start vLLM"
  bash "${START_VLLM_SCRIPT}" 2>&1 | tee "${LOG_DIR}/start_vllm.log"
  VLLM_PID="$(sed -n 's/.*PID: //p' "${LOG_DIR}/start_vllm.log" | tail -1 || true)"

  for _ in {1..60}; do
    if health_check; then
      echo "[pipeline] vLLM ready: ${VLLM_HEALTH_URL}"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: vLLM did not become ready: ${VLLM_HEALTH_URL}" >&2
  exit 3
}

latest_run_dir() {
  local root="$1"
  local latest=""
  shopt -s nullglob
  for dir in "${root}"/*; do
    if [[ -d "${dir}" ]]; then
      latest="${dir}"
    fi
  done
  shopt -u nullglob
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  ls -td "${root}"/*/ 2>/dev/null | head -1
}

copy_report_bundle() {
  local bench_name="$1"
  local bench_root="$2"
  local latest
  latest="$(latest_run_dir "${bench_root}")"
  mkdir -p "${REPORT_DIR}/${bench_name}"
  cp -a "${latest%/}/." "${REPORT_DIR}/${bench_name}/"
  if [[ -f "${latest%/}/evaluation_report.md" ]]; then
    cp "${latest%/}/evaluation_report.md" "${REPORT_DIR}/${bench_name}_evaluation_report.md"
  fi
  echo "[pipeline] ${bench_name} output: ${latest%/}"
  echo "[pipeline] ${bench_name} report bundle: ${REPORT_DIR}/${bench_name}"
}

run_contact() {
  log_step "Contact Rubric Bench"
  OUTPUT="${CONTACT_OUTPUT}" \
  FAIL_ON_GENERATION_ERROR="${FAIL_ON_GENERATION_ERROR:-1}" \
  CANDIDATE_MAX_OUTPUT_TOKENS="${CONTACT_MAX_OUTPUT_TOKENS:-2048}" \
  bash "${CONTACT_SCRIPT}" 2>&1 | tee "${LOG_DIR}/contact_rubric.log"
  copy_report_bundle "contact_rubric" "${CONTACT_OUTPUT}"
}

run_itbench() {
  log_step "ITBench"
  (
    cd "${ITBENCH_ROOT}"
    INPUT_FILE="${ITBENCH_INPUT_FILE}" \
    RULES_FILE="${ITBENCH_RULES_FILE}" \
    OUTPUT="${ITBENCH_OUTPUT}" \
    bash "${ITBENCH_SCRIPT}"
  ) 2>&1 | tee "${LOG_DIR}/itbench.log"
  copy_report_bundle "itbench" "${ITBENCH_OUTPUT}"
}

run_leadbench() {
  log_step "LeadBench"
  (
    cd "${LEADBENCH_ROOT}"
    INPUT_FILE="${LEADBENCH_INPUT_FILE}" \
    RULES_FILE="${LEADBENCH_RULES_FILE}" \
    OUTPUT="${LEADBENCH_OUTPUT}" \
    "${EVAL_PYTHON}" "${LEADBENCH_SCRIPT}"
  ) 2>&1 | tee "${LOG_DIR}/leadbench.log"
  copy_report_bundle "leadbench" "${LEADBENCH_OUTPUT}"
}

run_leadbench_excellent() {
  log_step "LeadBench Excellent"
  if [[ ! -d "${LEADBENCH_EXCELLENT_PACKAGE}" ]]; then
    echo "ERROR: leadbench_excellent package not found: ${LEADBENCH_EXCELLENT_PACKAGE}" >&2
    exit 4
  fi
  (
    cd "${LEADBENCH_EXCELLENT_ROOT}"
    INPUT_FILE="${LEADBENCH_EXCELLENT_INPUT_FILE}" \
    RULES_FILE="${LEADBENCH_EXCELLENT_RULES_FILE}" \
    OUTPUT_FILE="${LEADBENCH_EXCELLENT_OUTPUT}/evaluation_results.jsonl" \
    PYTHONPATH="${LEADBENCH_EXCELLENT_ROOT}:${PYTHONPATH:-}" \
    "${EVAL_PYTHON}" "${LEADBENCH_EXCELLENT_SCRIPT}"
  ) 2>&1 | tee "${LOG_DIR}/leadbench_excellent.log"
  copy_report_bundle "leadbench_excellent" "${LEADBENCH_EXCELLENT_OUTPUT}"
}

write_manifest() {
  cat > "${RUN_DIR}/manifest.json" <<JSON
{
  "timestamp": "${TIMESTAMP}",
  "run_dir": "${RUN_DIR}",
  "candidate_model": "${CANDIDATE_MODEL_NAME}",
  "candidate_api_base": "${CANDIDATE_API_BASE}",
  "judge_model": "${JUDGE_MODEL_NAME}",
  "judge_api_base": "${JUDGE_API_BASE}",
  "concurrency": "${CONCURRENCY}",
  "evaluation_limit": "${EVALUATION_LIMIT}",
  "contact_output": "${CONTACT_OUTPUT}",
  "itbench_output": "${ITBENCH_OUTPUT}",
  "leadbench_output": "${LEADBENCH_OUTPUT}",
  "leadbench_excellent_root": "${LEADBENCH_EXCELLENT_ROOT}",
  "leadbench_excellent_package": "${LEADBENCH_EXCELLENT_PACKAGE}",
  "leadbench_excellent_output": "${LEADBENCH_EXCELLENT_OUTPUT}",
  "reports": "${REPORT_DIR}"
}
JSON
}

write_summary() {
  {
    echo "# Benchmark Pipeline Summary"
    echo
    echo "- Run Dir: \`${RUN_DIR}\`"
    echo "- Candidate Model: \`${CANDIDATE_MODEL_NAME}\`"
    echo "- Candidate API: \`${CANDIDATE_API_BASE}\`"
    echo "- Judge Model: \`${JUDGE_MODEL_NAME}\`"
    echo "- Judge API: \`${JUDGE_API_BASE}\`"
    echo
    echo "## Reports"
    for report in "${REPORT_DIR}"/*_evaluation_report.md; do
      [[ -f "${report}" ]] || continue
      echo "- [$(basename "${report}")](${report})"
    done
  } > "${RUN_DIR}/summary.md"
}

write_manifest
ensure_vllm

if [[ "${RUN_CONTACT}" == "1" ]]; then
  run_contact
fi
if [[ "${RUN_ITBENCH}" == "1" ]]; then
  run_itbench
fi
if [[ "${RUN_LEADBENCH}" == "1" ]]; then
  run_leadbench
fi
if [[ "${RUN_LEADBENCH_EXCELLENT}" == "1" ]]; then
  run_leadbench_excellent
fi

write_summary

echo
echo "[pipeline] all requested benches finished"
echo "[pipeline] run_dir=${RUN_DIR}"
echo "[pipeline] summary=${RUN_DIR}/summary.md"
echo "[pipeline] reports=${REPORT_DIR}"
