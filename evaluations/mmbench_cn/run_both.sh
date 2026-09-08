#!/usr/bin/env bash
# 顺序跑 Qwen3.5-2B 和 Qwen3.5-4B 的 MMBench-CN 评测。
# 用法: nohup bash evaluations/mmbench_cn/run_both.sh > ~/mmbench_cn_results/run_both.log 2>&1 &
set -uo pipefail

export QWEN35_2B_PATH="${QWEN35_2B_PATH:-/home/jxy/TTL/Qwen3.5-2B}"
export QWEN35_4B_PATH="${QWEN35_4B_PATH:-/home/jxy/TTL/Qwen3.5-4B}"

echo "===== [$(date)] Qwen3.5-2B 开始 ====="
MODEL_PATH="$QWEN35_2B_PATH" bash "$(dirname "$0")/run_mmbench_cn.sh"
echo "===== [$(date)] Qwen3.5-2B 结束，退出码 $? ====="

echo "===== [$(date)] Qwen3.5-4B 开始 ====="
MODEL_PATH="$QWEN35_4B_PATH" bash "$(dirname "$0")/run_mmbench_cn.sh"
echo "===== [$(date)] Qwen3.5-4B 结束，退出码 $? ====="

echo "===== [$(date)] 全部完成 ====="
