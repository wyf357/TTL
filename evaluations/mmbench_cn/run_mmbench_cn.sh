#!/usr/bin/env bash
# 用 lmms-eval 跑 MMBench-CN（dev split，可直接算分），模型为本地 Qwen3.5 权重。
# 用法:
#   MODEL_PATH=/path/to/Qwen3.5-2B bash evaluations/mmbench_cn/run_mmbench_cn.sh
# 可选环境变量: TASKS(默认 mmbench_cn_dev) BATCH_SIZE NUM_PROCESSES OUTPUT_PATH MAX_PIXELS LOG_SAMPLES
set -euo pipefail

: "${MODEL_PATH:?设置 MODEL_PATH 为本地模型目录}"

PY="${MM_PY:-$HOME/miniconda3/envs/mmbench/bin/python}"
ACCEL="$(dirname "$PY")/accelerate"
MODEL="${LMMS_MODEL:-qwen3_5}"
TASKS="${TASKS:-mmbench_cn_dev}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PORT="${MAIN_PORT:-29500}"
TAG="$(basename "$MODEL_PATH")"
OUTPUT_PATH="${OUTPUT_PATH:-$HOME/mmbench_cn_results/${TAG}}"
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
  --gen_kwargs "temperature=0,max_new_tokens=${MAX_NEW_TOKENS:-1024}"
)

if [[ "${LOG_SAMPLES:-0}" == "1" ]]; then
  CMD+=(--log_samples "--log_samples_suffix=${LOG_SAMPLES_SUFFIX:-mmbench_cn}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
