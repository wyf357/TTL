#!/usr/bin/env bash
# AutoDL 上依次评测三份本地多模态权重（需按实际架构设置 LMMS_MODEL_*）。
# 使用前请先: python ../../download_mmmu_hf_mirror.py --preset lmms-eval
# 并 pip install -r ../../requirements-mmmu-eval.txt
set -euo pipefail

ROOT_TMP="${ROOT_TMP:-/root/autodl-tmp}"
export HF_HOME="${HF_HOME:-${ROOT_TMP}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

run_one() {
  local name="$1"
  local model="$2"
  local path="$3"
  local out="${4}"
  echo "========== ${name} =========="
  LMMS_MODEL="${model}" \
  MODEL_PATH="${path}" \
  OUTPUT_PATH="${out}" \
  NUM_PROCESSES="${NUM_PROCESSES:-1}" \
  bash "$(dirname "$0")/run_mmmu_baseline.sh"
}

# 默认路径与模型封装名：请按 checkpoint 的 config.json / lmms-eval 文档调整
# Qwen3.x 多模态常见为 qwen3_vl；Qwen2.5-VL 为 qwen2_5_vl；Gemma 多模态常为 gemma3
run_one "qwen35-2b" "${LMMS_MODEL_QWEN35_2B:-qwen3_vl}" "${ROOT_TMP}/Qwen3.5-2B" "${ROOT_TMP}/mmmu_logs/qwen35_2b"
run_one "qwen35-4b" "${LMMS_MODEL_QWEN35_4B:-qwen3_vl}" "${ROOT_TMP}/Qwen3.5-4B" "${ROOT_TMP}/mmmu_logs/qwen35_4b"
run_one "gemma-e2b" "${LMMS_MODEL_GEMMA:-gemma3}" "${ROOT_TMP}/gemma_E2B" "${ROOT_TMP}/mmmu_logs/gemma_e2b"

echo "全部完成。日志目录: ${ROOT_TMP}/mmmu_logs/"
