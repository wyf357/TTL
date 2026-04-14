#!/usr/bin/env bash
# 从 Hugging Face 下载数据集 EmbodiedBench/EB-ALFRED。
#
# 国内镜像（推荐）：
#   export HF_ENDPOINT=https://hf-mirror.com
#
# 下载到临时目录（默认即 /tmp 下 json_2.1.0 根）：
#   export EB_DEST=/tmp/eb_alfred_json_2.1.0
#
# 仅下载 splits 中的一条任务（与 base 的第 0 条一致，便于先跑通）：
#   export EB_DOWNLOAD_MODE=single
#   export EB_EVAL_SET=base          # 可选，默认 base
#   export EB_TASK_INDEX=0           # 可选，默认 0
#
# 下载后把目录链到 EmbodiedBench 期望路径（会替换已有 json_2.1.0 符号链接）：
#   export EB_SYMLINK=1
#
# 用法：
#   export HF_ENDPOINT=https://hf-mirror.com
#   export EMBODIEDBENCH_SRC=/path/to/EmbodiedBench   # 默认 third_party/EmbodiedBench
#   export EMBENCH_VENV=/path/to/embench-venv
#   bash scripts/download_eb_alfred_dataset.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBENCH_VENV:=/root/autodl-tmp/conda-envs/embench}"
PY="${EMBENCH_VENV}/bin/python"
"$PY" -m pip install -q huggingface_hub
# hf_transfer 与部分镜像/代理不兼容时，可 export HF_HUB_ENABLE_HF_TRANSFER=0
exec "$PY" "$ROOT/scripts/download_eb_alfred_hf.py" "$@"
