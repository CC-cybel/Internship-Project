#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:=~/datasets/teacher_rollout_prompt_pool_public}"
: "${EXCLUDE_OPD:=~/datasets/opd_prompt_pool_50k/opd_prompt_pool_50k_clean.jsonl}"
: "${SEED:=42}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${MAX_SCAN_MULTIPLIER:=120}"
: "${CONDA_ENV:=llama_factory}"

echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] EXCLUDE_OPD=${EXCLUDE_OPD}"
echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[INFO] MAX_SCAN_MULTIPLIER=${MAX_SCAN_MULTIPLIER}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"

export HF_ENDPOINT

conda run -n "${CONDA_ENV}" python data_remake/download_teacher_rollout_prompt_pool.py \
  --output-dir "${OUT_DIR}" \
  --exclude-opd "${EXCLUDE_OPD}" \
  --seed "${SEED}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  --max-scan-multiplier "${MAX_SCAN_MULTIPLIER}" \
  "$@"

