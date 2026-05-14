#!/usr/bin/env bash
set -euo pipefail

# Example:
#   SWANLAB_API_KEY=xxx \
#   REWARD_ROUTER_ADDRESS=10.0.0.2:9000 \
#   JUDGE_MODEL=gpt-5.2 \
#   LEAD_JUDGE_MODEL=gpt-5.2 \
#   INSTRUCTION_JUDGE_MODEL=gpt-5.2 \
#   TOTAL_EPOCHS=3 \
#   ROLLOUT_EVERY_N_EPOCHS=1 \
#   bash recipe/single_turn_reward/train_ppo_single_turn_reward_8gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${PYTHON_BIN:=python3}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

: "${MODEL_PATH:=/data/wangpf/project/LlamaFactory/models/Qwen3-8B}"
: "${TRAIN_FILES:=/data/wangpf/project/data_remake/outputs/single_turn_rl_20k/single_turn_rl_20k.train.parquet}"
: "${VAL_FILES:=/data/wangpf/project/data_remake/outputs/single_turn_rl_20k/single_turn_rl_20k.val.parquet}"

: "${N_NODES:=1}"
: "${N_GPUS_PER_NODE:=8}"
: "${TOTAL_EPOCHS:=1}"
: "${TRAIN_BATCH_SIZE:=128}"
: "${MAX_PROMPT_LEN:=2048}"
: "${MAX_RESPONSE_LEN:=1024}"
: "${ROLLOUT_N:=4}"
: "${VAL_ROLLOUT_N:=1}"

: "${ACTOR_PPO_MINI_BSZ:=64}"
: "${ACTOR_PPO_MICRO_BSZ_PER_GPU:=2}"
: "${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:=12288}"
: "${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU:=2}"
: "${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:=12288}"

: "${LR:=1e-6}"
: "${ROLLOUT_TEMP:=1.0}"
: "${ROLLOUT_TOP_P:=1.0}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.7}"

: "${PROJECT_NAME:=verl_single_turn_rl}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
: "${EXP_NAME:=hard_dual_ppo_${RUN_TAG}}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/outputs/single_turn_ppo/${EXP_NAME}}"

: "${SAVE_FREQ:=100}"
: "${ROLLOUT_SAMPLE_COUNT:=8}"
: "${ROLLOUT_EVERY_N_EPOCHS:=1}"
: "${TEST_FREQ_STEPS:=}"

: "${REWARD_ROUTER_ADDRESS:=127.0.0.1:8000}"
: "${JUDGE_MODEL:=gpt-5.2}"
: "${LEAD_JUDGE_MODEL:=}"
: "${INSTRUCTION_JUDGE_MODEL:=}"
: "${INSTRUCTION_API_KEY:=}"
: "${API_KEY:=}"

: "${LEAD_WEIGHT:=0.4}"
: "${INSTRUCTION_WEIGHT:=0.4}"
: "${FORMAT_WEIGHT:=0.2}"
: "${HARD_FORMAT_GATE:=true}"
: "${FORMAT_GATE_THRESHOLD:=1.0}"
: "${FORMAT_FAIL_SCORE:=0.0}"

: "${LEAD_RULES_FILE:=/data/wangpf/project/verl/recipe/single_turn_reward/talk_eval_rule.json}"
: "${INSTRUCTION_RUBRICS_PATH:=/data/wangpf/project/verl/recipe/single_turn_reward/rubrics_instruction_following.json}"

: "${COLLECT_GENRM_IO:=false}"
: "${GENRM_IO_INCLUDE_EXTRA_INFO:=false}"
: "${GENRM_IO_PATH:=${OUTPUT_ROOT}/genrm_io.jsonl}"

: "${SWANLAB_MODE:=cloud}"
: "${SWANLAB_LOG_DIR:=${OUTPUT_ROOT}/swanlog}"

RESUME_PATH="${1:-null}"

mkdir -p "${OUTPUT_ROOT}"
VALIDATION_DATA_DIR="${OUTPUT_ROOT}/validation_rollouts"
mkdir -p "${VALIDATION_DATA_DIR}"

get_parquet_rows() {
  local file="$1"
  "${PYTHON_BIN}" -c "import sys
try:
    import pyarrow.parquet as pq
except Exception:
    print(-1); raise SystemExit(0)
path = sys.argv[1]
try:
    print(pq.ParquetFile(path).metadata.num_rows)
except Exception:
    print(-1)" "${file}"
}

TEST_FREQ="${TEST_FREQ_STEPS:--1}"
if [[ -z "${TEST_FREQ_STEPS}" ]]; then
  if [[ "${ROLLOUT_EVERY_N_EPOCHS}" -gt 0 ]]; then
    TRAIN_ROWS="$(get_parquet_rows "${TRAIN_FILES}")"
    if [[ "${TRAIN_ROWS}" =~ ^[0-9]+$ ]] && [[ "${TRAIN_ROWS}" -gt 0 ]]; then
      STEPS_PER_EPOCH=$(( (TRAIN_ROWS + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
      TEST_FREQ=$(( STEPS_PER_EPOCH * ROLLOUT_EVERY_N_EPOCHS ))
      if [[ "${TEST_FREQ}" -lt 1 ]]; then
        TEST_FREQ=1
      fi
    fi
  fi
fi

export CUDA_VISIBLE_DEVICES
export SWANLAB_MODE
export SWANLAB_LOG_DIR

CMD=(
  "${PYTHON_BIN}" -m verl.trainer.main_ppo
  --config-path "${PROJECT_DIR}/recipe/single_turn_reward/config"
  --config-name ppo_single_turn_reward
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LEN}"
  "data.max_response_length=${MAX_RESPONSE_LEN}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BSZ}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ_PER_GPU}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMP}"
  "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXP_NAME}"
  "trainer.nnodes=${N_NODES}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.log_val_generations=${ROLLOUT_SAMPLE_COUNT}"
  "trainer.validation_data_dir=${VALIDATION_DATA_DIR}"
  "trainer.resume_from_path=${RESUME_PATH}"
  "+custom_reward_function.reward_kwargs.reward_router_address=${REWARD_ROUTER_ADDRESS}"
  "+custom_reward_function.reward_kwargs.judge_model=${JUDGE_MODEL}"
  "+custom_reward_function.reward_kwargs.lead_weight=${LEAD_WEIGHT}"
  "+custom_reward_function.reward_kwargs.instruction_weight=${INSTRUCTION_WEIGHT}"
  "+custom_reward_function.reward_kwargs.format_weight=${FORMAT_WEIGHT}"
  "+custom_reward_function.reward_kwargs.hard_format_gate=${HARD_FORMAT_GATE}"
  "+custom_reward_function.reward_kwargs.format_gate_threshold=${FORMAT_GATE_THRESHOLD}"
  "+custom_reward_function.reward_kwargs.format_fail_score=${FORMAT_FAIL_SCORE}"
  "+custom_reward_function.reward_kwargs.lead_rules_file=${LEAD_RULES_FILE}"
  "+custom_reward_function.reward_kwargs.instruction_rubrics_path=${INSTRUCTION_RUBRICS_PATH}"
  "+custom_reward_function.reward_kwargs.collect_genrm_io=${COLLECT_GENRM_IO}"
  "+custom_reward_function.reward_kwargs.genrm_io_path=${GENRM_IO_PATH}"
  "+custom_reward_function.reward_kwargs.genrm_io_include_extra_info=${GENRM_IO_INCLUDE_EXTRA_INFO}"
)

if [[ -n "${LEAD_JUDGE_MODEL}" ]]; then
  CMD+=("+custom_reward_function.reward_kwargs.lead_judge_model=${LEAD_JUDGE_MODEL}")
fi
if [[ -n "${INSTRUCTION_JUDGE_MODEL}" ]]; then
  CMD+=("+custom_reward_function.reward_kwargs.instruction_judge_model=${INSTRUCTION_JUDGE_MODEL}")
fi
if [[ -n "${API_KEY}" ]]; then
  CMD+=("+custom_reward_function.reward_kwargs.api_key=${API_KEY}")
fi
if [[ -n "${INSTRUCTION_API_KEY}" ]]; then
  CMD+=("+custom_reward_function.reward_kwargs.instruction_api_key=${INSTRUCTION_API_KEY}")
fi

echo "[INFO] project_dir=${PROJECT_DIR}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] test_freq=${TEST_FREQ} (steps)"
echo "[INFO] validation_data_dir=${VALIDATION_DATA_DIR}"
echo "[INFO] running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

