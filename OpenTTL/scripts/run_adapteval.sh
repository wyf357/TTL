#!/usr/bin/env bash
# AdaptEval：HF 推理 + TLM/Tent/EATA/COME 在线 TTA + 评测指标
#
# Usage:
#   bash scripts/run_adapteval.sh
#   bash scripts/run_adapteval.sh strategy=tent data.subset=gsm8k max_samples=50
#   bash scripts/run_adapteval.sh strategy=tlm online.enabled=true model.peft.enabled=true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TTL_ROOT="$(cd "$ROOT/.." && pwd)"

: "${ADAPTEVAL_DATA:="$TTL_ROOT/data/AdaptEval"}"
: "${ADAPTEVAL_GPU:=0}"

export CUDA_VISIBLE_DEVICES="$ADAPTEVAL_GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -d "$ADAPTEVAL_DATA" ]; then
    echo "AdaptEval 数据未找到: $ADAPTEVAL_DATA" >&2
    echo "请先运行: python $TTL_ROOT/download_adapteval_hf_mirror.py" >&2
    exit 1
fi

cd "$ROOT"
CMD=(python evaluations/run_adapteval.py)
CMD+=("data.local_root=$ADAPTEVAL_DATA" "inference.backend=hf")
if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "Running: ${CMD[*]}" >&2
exec "${CMD[@]}"
