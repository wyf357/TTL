#!/usr/bin/env bash
# 从 ModelScope 下载 Qwen3.5-2B，再跑 BFCL simple 评测。
#
# Usage:
#   bash scripts/run_bfcl_qwen35_download_and_eval.sh
#   bash scripts/run_bfcl_qwen35_download_and_eval.sh max_samples=20
set -euxo pipefail

TTL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$TTL_ROOT/OpenTTL"
MODEL_DIR="$TTL_ROOT/Qwen3.5-2B"
BFCL_DATA="$TTL_ROOT/data/bfcl"

PY_MS="${PY_MS:-/home/jxy/miniconda3/bin/python3}"
PY_TT="${PY_TT:-/home/jxy/miniconda3/envs/openttl/bin/python}"
if [ ! -x "$PY_MS" ]; then
    PY_MS="$(command -v python3)"
fi
if [ ! -x "$PY_TT" ]; then
    PY_TT="$PY_MS"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${BFCL_GPU:-0}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/4] Download Qwen3.5-2B from ModelScope -> $MODEL_DIR"
"$PY_MS" "$TTL_ROOT/download_qwen35_2b_modelscope.py" --local-dir "$MODEL_DIR"

if ! compgen -G "$MODEL_DIR/*.safetensors" > /dev/null; then
    echo "ERROR: 下载后仍未找到 $MODEL_DIR/*.safetensors" >&2
    ls -lah "$MODEL_DIR" >&2 || true
    exit 1
fi
ls -lh "$MODEL_DIR"/*.safetensors

echo "[2/4] Download BFCL v3 json -> $BFCL_DATA"
"$PY_MS" "$ROOT/scripts/download_bfcl_dataset.py" --out "$BFCL_DATA" \
    --categories simple multiple parallel parallel_multiple live_simple live_multiple irrelevance

echo "[3/4] BFCL parser smoke test"
"$PY_TT" "$ROOT/test_bfcl_eval.py"

echo "[4/4] Run BFCL eval (Qwen3.5-2B)"
cd "$ROOT"
CMD=(
    "$PY_TT" evaluations/run_bfcl.py
    model=qwen35_2b
    "model.pretrained_model_name_or_path=$MODEL_DIR"
    "bfcl_local_root=$BFCL_DATA"
    category=simple
)
if [ "$#" -gt 0 ]; then
    CMD+=("$@")
else
    CMD+=(max_samples=20)
fi
echo "Running: ${CMD[*]}" >&2
exec "${CMD[@]}"
