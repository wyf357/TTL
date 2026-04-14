#!/usr/bin/env bash
# 使用数据盘上的 embench venv 运行 OpenTTL 的 EmbodiedBench 桥接（需已配置 PYTHONPATH 与数据）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBODIEDBENCH_SRC:="$ROOT/third_party/EmbodiedBench"}"
: "${EMBENCH_VENV:=/root/autodl-tmp/conda-envs/embench}"
export PYTHONPATH="${EMBODIEDBENCH_SRC}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$EMBENCH_VENV/bin/python" "$ROOT/evaluations/run_embodiedbench.py" "$@"
