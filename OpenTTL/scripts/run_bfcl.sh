#!/usr/bin/env bash
# 使用本地 Qwen 模型跑 BFCL（Berkeley Function Calling Leaderboard）评测。
#
# Usage:
#   bash scripts/run_bfcl.sh
#   bash scripts/run_bfcl.sh category=simple max_samples=20
#   bash scripts/run_bfcl.sh category=multiple model=qwen35
#   BFCL_MODEL_PATH=/path/to/Qwen3.5-2B bash scripts/run_bfcl.sh
#
# Online TTA（边测边适应；默认 config 关闭，需显式开启）:
#   bash scripts/run_bfcl.sh --config-name=eval_bfcl_online category=simple max_samples=20
#   bash scripts/run_bfcl.sh --config-name=eval_bfcl_online strategy=tlm category=simple
#   bash scripts/run_bfcl.sh online.enabled=true model.peft.enabled=true strategy=tent
#
# 产物：outputs/bfcl_metrics.json、outputs/bfcl_result.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TTL_ROOT="$(cd "$ROOT/.." && pwd)"

: "${BFCL_GPU:=0}"
: "${BFCL_DATA:="$TTL_ROOT/data/bfcl"}"
: "${BFCL_MODEL_PATH:=""}"
: "${PY_TT:=/home/jxy/miniconda3/envs/openttl/bin/python}"

export CUDA_VISIBLE_DEVICES="$BFCL_GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ ! -x "$PY_TT" ]; then
    PY_TT="$(command -v python3)"
fi

resolve_model_path() {
    local candidates=()
    if [ -n "${BFCL_MODEL_PATH}" ]; then
        candidates+=("${BFCL_MODEL_PATH}")
    fi
    candidates+=(
        "/root/autodl-tmp/Qwen3.5-2B"
        "$TTL_ROOT/Qwen3.5-2B"
        "/root/autodl-tmp/Qwen3.5-4B"
        "$TTL_ROOT/Qwen3.5-4B"
    )
    local p
    for p in "${candidates[@]}"; do
        if [ -d "$p" ] && { [ -f "$p/config.json" ]; }; then
            if compgen -G "$p/*.safetensors" > /dev/null \
                || [ -f "$p/pytorch_model.bin" ] \
                || [ -f "$p/model.safetensors" ]; then
                echo "$p"
                return 0
            fi
        fi
    done
    return 1
}

MODEL_PATH="$(resolve_model_path || true)"
if [ -z "${MODEL_PATH}" ]; then
    echo "未找到带权重的本地 Qwen 模型。" >&2
    echo "请先下载，例如：" >&2
    echo "  python $TTL_ROOT/download_qwen35_2b_modelscope.py --local-dir $TTL_ROOT/Qwen3.5-2B" >&2
    echo "或设置 BFCL_MODEL_PATH=/path/to/model" >&2
    exit 1
fi

if [ ! -d "$BFCL_DATA" ] || [ -z "$(ls -A "$BFCL_DATA"/BFCL_v3_*.json 2>/dev/null || true)" ]; then
    echo "BFCL 数据未找到，正在下载到 $BFCL_DATA ..." >&2
    "$PY_TT" "$ROOT/scripts/download_bfcl_dataset.py" --out "$BFCL_DATA" \
        --categories simple multiple parallel parallel_multiple live_simple live_multiple irrelevance
fi

echo "========================================" >&2
echo "BFCL Benchmark Evaluation" >&2
echo "========================================" >&2
echo "Model:  $MODEL_PATH" >&2
echo "Data:   $BFCL_DATA" >&2
echo "GPU:    $BFCL_GPU" >&2
echo "Python: $PY_TT" >&2
echo "========================================" >&2

cd "$ROOT"
CMD=("$PY_TT" evaluations/run_bfcl.py)
CMD+=("model=qwen35_2b")
CMD+=("model.pretrained_model_name_or_path=$MODEL_PATH")
CMD+=("bfcl_local_root=$BFCL_DATA")

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "Running: ${CMD[*]}" >&2
exec "${CMD[@]}"
