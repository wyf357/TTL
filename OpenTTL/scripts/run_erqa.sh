#!/usr/bin/env bash
# Run ERQA Benchmark Evaluation with local Qwen3.5-VL model
#
# Usage:
#   bash scripts/run_erqa.sh
#   bash scripts/run_erqa.sh max_examples=10
#   bash scripts/run_erqa.sh tfrecord_path=/path/to/erqa.tfrecord model_path=/path/to/model
#
# Prerequisites:
#   1. Download ERQA dataset (erqa.tfrecord)
#   2. Download Qwen3.5-2B model to autodl-tmp
#   3. Install dependencies: tensorflow, transformers, torch, PIL

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration with defaults
: "${ERQA_TFRECORD:="$ROOT/data/erqa.tfrecord"}"
: "${ERQA_MODEL_PATH:="/root/autodl-tmp/Qwen3.5-2B"}"
: "${ERQA_MAX_EXAMPLES:=""}"
: "${ERQA_GPU:=0}"

# GPU Configuration
export CUDA_VISIBLE_DEVICES="$ERQA_GPU"
echo "Using GPU: $ERQA_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2

# Set PYTHONPATH to include OpenTTL source
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"

# Check if TFRecord file exists
if [ ! -f "$ERQA_TFRECORD" ]; then
    echo "Warning: ERQA TFRecord file not found at $ERQA_TFRECORD" >&2
    echo "Please download the ERQA dataset first:" >&2
    echo "  mkdir -p $ROOT/data" >&2
    echo "  # Download erqa.tfrecord from https://github.com/embodiedreasoning/ERQA" >&2
    echo "  cp /path/to/erqa.tfrecord $ROOT/data/" >&2
    exit 1
fi

# Check if model directory exists
if [ ! -d "$ERQA_MODEL_PATH" ]; then
    echo "Warning: Model directory not found at $ERQA_MODEL_PATH" >&2
    echo "Please download the Qwen3.5-2B model first:" >&2
    echo "  # Download from ModelScope or HuggingFace" >&2
    echo "  python download_qwen35_2b_modelscope.py" >&2
    exit 1
fi

echo "========================================" >&2
echo "ERQA Benchmark Evaluation" >&2
echo "========================================" >&2
echo "TFRecord: $ERQA_TFRECORD" >&2
echo "Model: $ERQA_MODEL_PATH" >&2
if [ -n "$ERQA_MAX_EXAMPLES" ]; then
    echo "Max Examples: $ERQA_MAX_EXAMPLES" >&2
else
    echo "Max Examples: All" >&2
fi
echo "========================================" >&2

# Build command
CMD="python $ROOT/evaluations/run_erqa.py"
CMD+=" tfrecord_path=$ERQA_TFRECORD"
CMD+=" model_path=$ERQA_MODEL_PATH"

if [ -n "$ERQA_MAX_EXAMPLES" ]; then
    CMD+=" max_examples=$ERQA_MAX_EXAMPLES"
fi

# Add any additional Hydra overrides passed as arguments
if [ $# -gt 0 ]; then
    CMD+=" $@"
fi

# Run evaluation
echo "Starting evaluation..." >&2
echo "Command: $CMD" >&2
echo "========================================" >&2

exec $CMD
