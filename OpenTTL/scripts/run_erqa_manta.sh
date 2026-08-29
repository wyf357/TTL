#!/usr/bin/env bash
# Run ERQA Benchmark Evaluation with MANTA strategy
#
# Usage:
#   bash scripts/run_erqa_manta.sh
#   bash scripts/run_erqa_manta.sh max_examples=10
#   bash scripts/run_erqa_manta.sh strategy.num_adapt_layers=2
#
# Prerequisites:
#   1. ERQA dataset (erqa.tfrecord)
#   2. Qwen3.5 model downloaded
#   3. Dependencies installed

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration with defaults
: "${ERQA_TFRECORD:="$ROOT/data/erqa.tfrecord"}"
: "${ERQA_MODEL:="qwen35_2b"}"
: "${ERQA_MAX_EXAMPLES:=""}"
: "${ERQA_GPU:=0}"

# GPU Configuration
export CUDA_VISIBLE_DEVICES="$ERQA_GPU"

echo "========================================"
echo "ERQA Benchmark Evaluation with MANTA"
echo "========================================"
echo "TFRecord: $ERQA_TFRECORD"
echo "Model: $ERQA_MODEL"
echo "Strategy: manta"
echo "Max Examples: ${ERQA_MAX_EXAMPLES:-All}"
echo "GPU: $ERQA_GPU"
echo "========================================"

# Build command
CMD="python $ROOT/evaluations/run_erqa.py"
CMD="$CMD tfrecord_path=$ERQA_TFRECORD"
CMD="$CMD model=$ERQA_MODEL"
CMD="$CMD strategy=manta"

if [ -n "$ERQA_MAX_EXAMPLES" ]; then
    CMD="$CMD max_examples=$ERQA_MAX_EXAMPLES"
fi

# Add any additional arguments
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

echo "Starting evaluation with MANTA strategy..."
echo "Command: $CMD"
echo "========================================"

# Run evaluation
cd "$ROOT"
eval $CMD
