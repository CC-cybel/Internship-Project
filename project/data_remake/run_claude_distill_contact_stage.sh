#!/bin/bash
# Claude distill 套联阶段小范围测试脚本

cd /data/chengch/project/data_remake

# 参考 itbench/run_eval.sh：本机/内网服务不走代理。
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"

# 这个 yunwu/Claude 接口在当前 127.0.0.1:2125 转发代理下会 TLS 断开；
# 直连已验证可用，因此该任务默认清掉外部代理环境。
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

PYTHON=/data/chengch/.conda/envs/verl/bin/python

INPUT=/data/chengch/project/data_remake/intermediate/opd_offline_full_v3_a2r8_train.jsonl
OUTPUT=/data/chengch/project/data_remake/outputs/claude_distlill.jsonl
RAW_LOG=/data/chengch/project/data_remake/logs/claude_distlill_raw.txt

echo "开始 Claude distill 小范围测试..."
echo "输入: ${INPUT}"
echo "输出: ${OUTPUT}"
echo "NO_PROXY: ${NO_PROXY}"
echo "外部 API 代理: disabled"
echo "模型: claude-sonnet-4-5-20250929"

${PYTHON} /data/chengch/project/data_remake/claude_distill_contact_stage.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --raw-log "${RAW_LOG}" \
  --limit 5 \
  --max-workers 5 \
  --timeout 300 \
  --max-tokens 2048

echo "Claude distill 测试完成。"
