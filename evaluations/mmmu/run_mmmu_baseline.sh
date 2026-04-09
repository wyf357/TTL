#!/usr/bin/env bash
# 使用 lmms-eval 对本地多模态权重跑 MMMU（无 TTA，仅推理）。
# 用法见同目录 README.md。
set -euo pipefail

: "${LMMS_MODEL:?设置 LMMS_MODEL，例如 qwen3_vl / qwen2_5_vl / gemma3}"
: "${MODEL_PATH:?设置 MODEL_PATH 为本地模型目录}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

TASKS="${TASKS:-mmmu_val}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PORT="${MAIN_PORT:-29500}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/mmmu_baseline}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"
ATTN="${ATTN_IMPLEMENTATION:-sdpa}"
INTERLEAVE="${INTERLEAVE_VISUALS:-False}"

mkdir -p "${OUTPUT_PATH}"

# model_args 逗号分隔，勿随意加空格
MODEL_ARGS="pretrained=${MODEL_PATH},max_pixels=${MAX_PIXELS},attn_implementation=${ATTN},interleave_visuals=${INTERLEAVE}"

CMD=(accelerate launch "--num_processes=${NUM_PROCESSES}" "--main_process_port=${MAIN_PORT}" -m lmms_eval
  --model "${LMMS_MODEL}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ "${LOG_SAMPLES:-0}" == "1" ]]; then
  CMD+=(--log_samples "--log_samples_suffix=${LOG_SAMPLES_SUFFIX:-mmmu}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
