#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

deactivate 2>/dev/null || true

: "${PYTHON_BIN:=/data/chengch/.conda/envs/verl/bin/python}"
: "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"

: "${MODEL_PATH:=/data/chengch/normal_stage2_exp7_qwen3_8b_full_sft_t4}"
: "${CONTACT_TEACHER:=/data1/chengch/models/qwen3_8b_contact_step200}"
: "${MID_TEACHER:=/data1/chengch/models/qwen3_8b_normal_mid}"

: "${DATA_DIR:=/data/chengch/project/verl/recipe/opd_multi_teacher/v2/data}"
: "${TRAIN_FILE:=${DATA_DIR}/opd_multi_teacher_v2_teacher_generated.train.jsonl}"
: "${VAL_FILE:=${TRAIN_FILE}}"
: "${BUILD_DATA:=false}"
: "${SOURCE_FILE:=/data/chengch/project/verl/recipe/opd_multi_teacher/data/opd_multi_teacher_10k.train.jsonl}"

: "${TRAIN_BATCH_SIZE:=8}"
: "${MAX_PROMPT_LENGTH:=2040}"
: "${MAX_RESPONSE_LENGTH:=512}"
: "${BUILD_DATA_BATCH_SIZE:=16}"
: "${BUILD_DATA_GPU_MEMORY_UTIL:=0.45}"
: "${BUILD_DATA_MAX_MODEL_LEN:=4096}"
: "${BUILD_DATA_MAX_PROMPT_TOKENS:=${MAX_PROMPT_LENGTH}}"
: "${BUILD_DATA_MAX_NEW_TOKENS:=${MAX_RESPONSE_LENGTH}}"

: "${N_GPUS_PER_NODE:=4}"
: "${NNODES:=1}"
: "${TOTAL_EPOCHS:=2}"

: "${PPO_MINI_BATCH_SIZE:=4}"
: "${PPO_MICRO_BATCH_SIZE_PER_GPU:=1}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=8192}"

: "${ROLLOUT_N:=1}"
: "${ROLLOUT_TP:=1}"
: "${ROLLOUT_GPU_MEMORY_UTIL:=0.45}"
: "${ROLLOUT_MAX_BATCHED_TOKENS:=4096}"
: "${ROLLOUT_MAX_NUM_SEQS:=16}"
: "${ROLLOUT_ENFORCE_EAGER:=True}"
: "${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU:=1}"
: "${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:=8192}"

: "${TEACHER_TP:=1}"
: "${TEACHER_GPU_MEMORY_UTIL:=0.45}"
: "${TEACHER_MAX_BATCHED_TOKENS:=4096}"
: "${TEACHER_MAX_NUM_SEQS:=4}"
: "${TEACHER_ENFORCE_EAGER:=True}"

: "${DISTILL_TOPK:=64}"
: "${DISTILL_LOSS_MAX_CLAMP:=10.0}"
: "${DISTILL_LOGPROB_MIN_CLAMP:=-10.0}"
: "${LR:=2e-6}"
: "${SAVE_FREQ:=50}"
: "${TEST_FREQ:=50}"
: "${LOG_VAL_GENERATIONS:=1}"
: "${RESUME_MODE:=resume_path}"
: "${RESUME_FROM_PATH:=/data1/chengch/verl_outputs/opd_multi_teacher/v2/qwen3_8b_v2_forward_kl_topk_20260430_104558/global_step_150}"

: "${COLLECT_GENRM_IO:=True}"
: "${GENRM_IO_INCLUDE_EXTRA_INFO:=false}"

: "${PROJECT_NAME:=verl_opd_multi_teacher_v2}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
: "${EXP_NAME:=qwen3_8b_v2_forward_kl_topk_${RUN_TAG}}"
: "${OUTPUT_BASE_DIR:=/data1/chengch/verl_outputs/opd_multi_teacher/v2}"
: "${OUTPUT_DIR:=${OUTPUT_BASE_DIR}/${EXP_NAME}}"
: "${GENRM_IO_PATH:=${OUTPUT_DIR}/genrm_io.jsonl}"

: "${SWANLAB_API_KEY:=}"
: "${SWANLAB_MODE:=cloud}"
: "${SWANLAB_LOG_DIR:=${OUTPUT_DIR}/swanlog}"

mkdir -p "${OUTPUT_DIR}"
VALIDATION_DATA_DIR="${OUTPUT_DIR}/validation_rollouts"
mkdir -p "${VALIDATION_DATA_DIR}"

export CUDA_VISIBLE_DEVICES
export SWANLAB_API_KEY
export SWANLAB_MODE
export SWANLAB_LOG_DIR

if [[ "${BUILD_DATA}" == "true" || ! -f "${TRAIN_FILE}" ]]; then
  "${PYTHON_BIN}" recipe/opd_multi_teacher/v2/build_teacher_generated_data.py \
    --input "${SOURCE_FILE}" \
    --output-dir "${DATA_DIR}" \
    --contact-teacher "${CONTACT_TEACHER}" \
    --mid-teacher "${MID_TEACHER}" \
    --batch-size "${BUILD_DATA_BATCH_SIZE}" \
    --gpu-memory-utilization "${BUILD_DATA_GPU_MEMORY_UTIL}" \
    --max-model-len "${BUILD_DATA_MAX_MODEL_LEN}" \
    --max-prompt-tokens "${BUILD_DATA_MAX_PROMPT_TOKENS}" \
    --max-new-tokens "${BUILD_DATA_MAX_NEW_TOKENS}" \
    --tensor-parallel-size "${TEACHER_TP}"
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
  "data.shuffle=True"
  "data.seed=20260430"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
  "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.dtype=bfloat16"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=0.7"
  "actor_rollout_ref.rollout.top_p=0.9"
  "actor_rollout_ref.rollout.max_model_len=4096"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTIL}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/opd_multi_teacher/v2/teacher_forced_agent_loop.yaml"
  "actor_rollout_ref.rollout.agent.default_agent_loop=teacher_forced_agent"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16"
  "reward.reward_model.enable=False"
  "reward.reward_manager.name=naive"
  "reward.custom_reward_function.path=recipe/opd_multi_teacher/v2/reward_zero.py"
  "reward.custom_reward_function.name=compute_score"
  "+reward.custom_reward_function.reward_kwargs.collect_genrm_io=${COLLECT_GENRM_IO}"
  "+reward.custom_reward_function.reward_kwargs.genrm_io_path=${GENRM_IO_PATH}"
  "+reward.custom_reward_function.reward_kwargs.genrm_io_include_extra_info=${GENRM_IO_INCLUDE_EXTRA_INFO}"
  "distillation.enabled=True"
  "distillation.teacher_model.enable_resource_pool=False"
  "distillation.teacher_model.model_path=${CONTACT_TEACHER}"
  "distillation.teacher_model.inference.name=vllm"
  "distillation.teacher_model.inference.dtype=bfloat16"
  "distillation.teacher_model.inference.temperature=1.0"
  "distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}"
  "distillation.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTIL}"
  "distillation.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_BATCHED_TOKENS}"
  "distillation.teacher_model.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS}"
  "distillation.teacher_model.inference.max_model_len=4096"
  "distillation.teacher_model.inference.enforce_eager=${TEACHER_ENFORCE_EAGER}"
  "distillation.teacher_model.inference.free_cache_engine=True"
  "distillation.distillation_loss.loss_mode=forward_kl_topk"
  "distillation.distillation_loss.topk=${DISTILL_TOPK}"
  "distillation.distillation_loss.use_policy_gradient=False"
  "distillation.distillation_loss.use_task_rewards=False"
  "distillation.distillation_loss.distillation_loss_coef=1.0"
  "distillation.distillation_loss.loss_max_clamp=${DISTILL_LOSS_MAX_CLAMP}"
  "distillation.distillation_loss.log_prob_min_clamp=${DISTILL_LOGPROB_MIN_CLAMP}"
  "+distillation.teacher_routes=[{name:contact,model_path:${CONTACT_TEACHER}},{name:mid,model_path:${MID_TEACHER}}]"
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
  "trainer.logger=[\"console\",\"swanlab\"]"
)

CMD+=("trainer.resume_mode=${RESUME_MODE}")
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  CMD+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] student=${MODEL_PATH}"
echo "[INFO] contact_teacher=${CONTACT_TEACHER}"
echo "[INFO] mid_teacher=${MID_TEACHER}"
echo "[INFO] train_file=${TRAIN_FILE}"
echo "[INFO] val_file=${VAL_FILE}"
echo "[INFO] distill=forward_kl_topk, topk=${DISTILL_TOPK}, teacher_forced_agent=true"
echo "[INFO] swanlab_mode=${SWANLAB_MODE}, swanlab_log_dir=${SWANLAB_LOG_DIR}"
echo "[INFO] log_val_generations=${LOG_VAL_GENERATIONS}, validation_data_dir=${VALIDATION_DATA_DIR}"
echo "[INFO] collect_genrm_io=${COLLECT_GENRM_IO}, genrm_io_path=${GENRM_IO_PATH}"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
