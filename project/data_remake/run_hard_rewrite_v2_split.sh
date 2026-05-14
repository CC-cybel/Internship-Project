#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/data/chengch/project/data_remake/rewrite_dialogues1_v2.py"
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME="deepseek-v4-flash"
MAX_WORKERS=40

PART1_INPUT="/data/chengch/project/data_remake/runs/hard_reverse_tongyi_v2_action_part1_0_10000.json"
PART1_OUTPUT="/data/chengch/project/data_remake/runs/hard_rewrite_v2_part1_0_10000.json"
PART1_CACHE="/data/chengch/project/data_remake/cache/hard_rewrite_v2_part1_0_10000"
PART1_LOG="/data/chengch/project/data_remake/logs/hard_rewrite_v2_part1_0_10000.txt"
PART1_API_KEY="sk-4159aabf91b84866af83a01d996e91ec"

PART2_INPUT="/data/chengch/project/data_remake/runs/hard_reverse_tongyi_v2_action_part2_10000_20000.json"
PART2_OUTPUT="/data/chengch/project/data_remake/runs/hard_rewrite_v2_part2_10000_20000.json"
PART2_CACHE="/data/chengch/project/data_remake/cache/hard_rewrite_v2_part2_10000_20000"
PART2_LOG="/data/chengch/project/data_remake/logs/hard_rewrite_v2_part2_10000_20000.txt"
PART2_API_KEY="sk-f76c711b79a24e358d6fa4ca4c69d670"

python "$SCRIPT" \
  --input "$PART1_INPUT" \
  --output "$PART1_OUTPUT" \
  --cache-dir "$PART1_CACHE" \
  --raw-log "$PART1_LOG" \
  --max-workers "$MAX_WORKERS" \
  --model "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --api-key "$PART1_API_KEY" &

PID1=$!

python "$SCRIPT" \
  --input "$PART2_INPUT" \
  --output "$PART2_OUTPUT" \
  --cache-dir "$PART2_CACHE" \
  --raw-log "$PART2_LOG" \
  --max-workers "$MAX_WORKERS" \
  --model "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --api-key "$PART2_API_KEY" &

PID2=$!

wait "$PID1"
wait "$PID2"

echo "✅ 两段 hard rewrite 已完成："
echo "  $PART1_OUTPUT"
echo "  $PART2_OUTPUT"
