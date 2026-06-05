#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

: "${PYTHON_BIN:=/data/chengch/.conda/envs/verl/bin/python}"

"${PYTHON_BIN}" recipe/dual_zscore_grpo/tests/test_dual_zscore_advantage.py
"${PYTHON_BIN}" recipe/dual_zscore_grpo/tests/alpha_sweep_toy.py --output-mode raw
"${PYTHON_BIN}" recipe/dual_zscore_grpo/tests/alpha_sweep_random.py --output-mode raw --trials 100
