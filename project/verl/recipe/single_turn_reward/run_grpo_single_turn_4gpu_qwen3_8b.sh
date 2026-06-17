#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# Drop inherited venv marker if present, but keep the active conda env.
deactivate 2>/dev/null || true

: "${PYTHON_BIN:=/data/chengch/.conda/envs/verl/bin/python}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"

: "${TONGYI_API_BASE:=https://dashscope.aliyuncs.com/compatible-mode/v1}"
: "${TONGYI_API_KEY:="your_api_key_here"}"
: "${JUDGE_MODEL:=qwen-plus}"

: "${MODEL_PATH:=/data/wangpf/project/LlamaFactory/saves/qwen3-8b/full/anti_hijack_abc_stage2}"
: "${TRAIN_FILE:=/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k/single_turn_rl_random_rounds.train.parquet}"
: "${VAL_FILE:=/data/chengch/project/rl_remake/outputs/single_turn_rl_random_rounds_20k/single_turn_rl_random_rounds.val.parquet}"

: "${TRAIN_BATCH_SIZE:=32}"
: "${MAX_PROMPT_LENGTH:=2048}"
: "${MAX_RESPONSE_LENGTH:=384}"

: "${N_GPUS_PER_NODE:=4}"
: "${NNODES:=1}"
: "${TOTAL_EPOCHS:=1}"

: "${PPO_MINI_BATCH_SIZE:=16}"
: "${PPO_MICRO_BATCH_SIZE_PER_GPU:=2}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=6144}"

: "${ROLLOUT_N:=2}"
: "${ROLLOUT_TP:=1}"
: "${ROLLOUT_GPU_MEMORY_UTIL:=0.55}"
: "${ROLLOUT_MAX_BATCHED_TOKENS:=4096}"
: "${ROLLOUT_MAX_NUM_SEQS:=256}"
: "${ROLLOUT_ENFORCE_EAGER:=True}"
: "${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU:=2}"
: "${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:=6144}"

: "${LR:=3e-6}"
: "${SAVE_FREQ:=50}"
: "${TEST_FREQ:=1}"
: "${LOG_VAL_GENERATIONS:=1}"

: "${PROJECT_NAME:=verl_grpo_single_turn}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
: "${EXP_NAME:=qwen3_8b_4gpu_stage1_first2_cloud_${RUN_TAG}}"
: "${OUTPUT_DIR:=${PROJECT_DIR}/outputs/grpo_single_turn/${EXP_NAME}}"

: "${SWANLAB_API_KEY:=ZqlCkcrue6FEBG24I91wi}"
: "${SWANLAB_MODE:=cloud}"
: "${SWANLAB_LOG_DIR:=${OUTPUT_DIR}/swanlog}"

mkdir -p "${OUTPUT_DIR}"
VALIDATION_DATA_DIR="${OUTPUT_DIR}/validation_rollouts"
mkdir -p "${VALIDATION_DATA_DIR}"

export CUDA_VISIBLE_DEVICES
export SWANLAB_API_KEY
export SWANLAB_MODE
export SWANLAB_LOG_DIR

if [[ -z "${TONGYI_API_KEY}" ]]; then
  echo "[ERROR] TONGYI_API_KEY is empty."
  echo "[ERROR] export TONGYI_API_KEY=your_api_key and rerun."
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -m verl.trainer.main_ppo
  --config-name ppo_trainer
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.001"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16"
  "actor_rollout_ref.rollout.dtype=bfloat16"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTIL}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16"
  "reward.reward_model.enable=False"
  "reward.reward_manager.name=naive"
  "reward.custom_reward_function.path=recipe/single_turn_reward/reward_function_first2_stage1_cloud.py"
  "reward.custom_reward_function.name=compute_score"
  "+reward.custom_reward_function.reward_kwargs.api_base=${TONGYI_API_BASE}"
  "+reward.custom_reward_function.reward_kwargs.api_key=${TONGYI_API_KEY}"
  "+reward.custom_reward_function.reward_kwargs.judge_model=${JUDGE_MODEL}"
  "+reward.custom_reward_function.reward_kwargs.enable_model_judge=True"
  "+reward.custom_reward_function.reward_kwargs.rule_weight=0"
  "+reward.custom_reward_function.reward_kwargs.model_weight=1.0"
  "+reward.custom_reward_function.reward_kwargs.stage_weight=0.35"
  "+reward.custom_reward_function.reward_kwargs.objective_weight=0.35"
  "+reward.custom_reward_function.reward_kwargs.professional_weight=0.20"
  "+reward.custom_reward_function.reward_kwargs.safety_weight=0.10"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXP_NAME}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
  "trainer.validation_data_dir=${VALIDATION_DATA_DIR}"
  "trainer.val_before_train=False"
  "trainer.logger=[\"console\",\"swanlab\",\"tensorboard\"]"
)

echo "[INFO] project_dir=${PROJECT_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] using model=${MODEL_PATH}"
echo "[INFO] using train=${TRAIN_FILE}"
echo "[INFO] using val=${VAL_FILE}"
echo "[INFO] rollout_n=${ROLLOUT_N}"
echo "[INFO] rollout_gpu_memory_util=${ROLLOUT_GPU_MEMORY_UTIL}"
echo "[INFO] rollout_max_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
echo "[INFO] rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "[INFO] test_freq=${TEST_FREQ}"
echo "[INFO] log_val_generations=${LOG_VAL_GENERATIONS}"
echo "[INFO] validation_data_dir=${VALIDATION_DATA_DIR}"
echo "[INFO] swanlab_mode=${SWANLAB_MODE}"
echo "[INFO] swanlab_log_dir=${SWANLAB_LOG_DIR}"
echo "[INFO] launching GRPO on ${N_GPUS_PER_NODE} GPUs"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
