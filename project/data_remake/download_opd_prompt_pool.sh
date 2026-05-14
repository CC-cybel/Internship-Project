#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:=~/datasets/opd_prompt_pool_50k}"
: "${SEED:=42}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${MAX_SCAN_MULTIPLIER:=80}"

echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[INFO] MAX_SCAN_MULTIPLIER=${MAX_SCAN_MULTIPLIER}"

export HF_ENDPOINT

python3 data_remake/download_opd_prompt_pool.py \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  --max-scan-multiplier "${MAX_SCAN_MULTIPLIER}" \
  "$@"

