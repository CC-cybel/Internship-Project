#!/bin/bash
# vLLM Server 启动脚本 - 2卡 (0,1)
# 模型: /data1/chengch/models/qwen3_8b_mid_short_step350

MODEL_PATH=/data1/chengch/models/qwen3_8b_normal_mid_140
PORT=8002
SERVED_MODEL_NAME=qwen3_8b_normal_mid_140
GPU_MEMORY_UTIL=0.85
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
GPU_IDS="2"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

VLLM_ENV=/data/wangpf/project/miniconda3/envs/qwen35-vllm
PYTHON=${VLLM_ENV}/bin/python
LOG_FILE=/data/chengch/lm-evaluation-harness/itbench-master/itbench-master/vllm_server.log

echo "启动 vLLM Server (2卡: 0,1)..."
echo "模型路径: ${MODEL_PATH}"
echo "端口: ${PORT}"
echo "Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
echo "日志文件: ${LOG_FILE}"

${PYTHON} -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype auto \
    --trust-remote-code \
    > ${LOG_FILE} 2>&1 &

echo "vLLM Server 已后台启动, PID: $!"
echo "等待服务就绪..."
sleep 5

# 等待服务就绪
for i in {1..30}; do
    if curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; then
        echo "vLLM Server 已就绪!"
        exit 0
    fi
    sleep 2
done

echo "警告: vLLM Server 可能尚未就绪，请检查日志: ${LOG_FILE}"
