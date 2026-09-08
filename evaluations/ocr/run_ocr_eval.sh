#!/usr/bin/env bash
# 用 lmms-eval 跑 OCR 类基准（OCRBench / OmniDocBench），模型为本地 Qwen3.5 权重。
# 用法:
#   TASKS=ocrbench      MODEL_PATH=/home/jxy/TTL/Qwen3.5-2B bash evaluations/ocr/run_ocr_eval.sh
#   TASKS=omnidocbench  MODEL_PATH=/home/jxy/TTL/Qwen3.5-2B bash evaluations/ocr/run_ocr_eval.sh
# 可选环境变量: OUTPUT_PATH BATCH_SIZE LIMIT(调试) MAX_NEW_TOKENS(默认不设，用任务自带值)
set -euo pipefail

: "${MODEL_PATH:?设置 MODEL_PATH 为本地模型目录}"
: "${TASKS:?设置 TASKS，如 ocrbench / omnidocbench}"

PY="${MM_PY:-$HOME/miniconda3/envs/mmbench/bin/python}"
ACCEL="$(dirname "$PY")/accelerate"
MODEL="${LMMS_MODEL:-qwen3_5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PORT="${MAIN_PORT:-29501}"
TAG="$(basename "$MODEL_PATH")"
OUTPUT_PATH="${OUTPUT_PATH:-$HOME/ocr_results/${TASKS}/${TAG}}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"
ATTN="${ATTN_IMPLEMENTATION:-sdpa}"
INTERLEAVE="${INTERLEAVE_VISUALS:-False}"

# HF 数据集走镜像（TUNA 不镜像 HF hub，hf-mirror 为国内可用镜像）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH},max_pixels=${MAX_PIXELS},attn_implementation=${ATTN},interleave_visuals=${INTERLEAVE},enable_thinking=False"

CMD=("${ACCEL}" launch "--num_processes=${NUM_PROCESSES}" "--main_process_port=${MAIN_PORT}" -m lmms_eval
  --model "${MODEL}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
)

# 不设 MAX_NEW_TOKENS 时用任务 yaml 自带的 generation_kwargs
# （OCRBench=128；OmniDocBench=4096，页面转 markdown 需要长输出，勿随意改小）
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  CMD+=(--gen_kwargs "temperature=0,max_new_tokens=${MAX_NEW_TOKENS}")
fi

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
