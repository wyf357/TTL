#!/usr/bin/env bash
# 顺序跑 Qwen3.5-2B 的 OCRBench 和 OmniDocBench 评测。
# 用法: nohup bash evaluations/ocr/run_ocr_both.sh > ~/ocr_results/run_both.log 2>&1 &
set -uo pipefail

export QWEN35_2B_PATH="${QWEN35_2B_PATH:-/home/jxy/TTL/Qwen3.5-2B}"
DIR="$(cd "$(dirname "$0")" && pwd)"

for task in ocrbench omnidocbench; do
  echo "===== [$(date)] ${task} 开始 ====="
  TASKS="$task" MODEL_PATH="$QWEN35_2B_PATH" bash "$DIR/run_ocr_eval.sh"
  echo "===== [$(date)] ${task} 结束，退出码 $? ====="
done
echo "===== [$(date)] 全部完成 ====="
