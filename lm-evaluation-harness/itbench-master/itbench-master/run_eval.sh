#!/bin/bash
# itbench 评测脚本
# 评测模型: /data1/chengch/models/qwen3_8b_mid_short_step350
# 领域: Psychiatry
# 评测数量: 100条

cd /data/chengch/lm-evaluation-harness/itbench-master/itbench-master

export NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"

EVAL_ENV=/data/wangpf/project/miniconda3/envs/lm-evaluation-harness
PYTHON=${EVAL_ENV}/bin/python

echo "开始 itbench 评测..."
echo "模型: qwen3_8b_mid_short_step350"
echo "领域: psychiatry"
echo "评测数量: 100"

${PYTHON} scripts/evaluate_golden_history.py

echo "评测完成!"
