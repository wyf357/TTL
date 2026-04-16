#!/usr/bin/env bash
# 从 Hugging Face 下载 EB-ALFRED 数据集（支持国内镜像）
#
# 用法：
#   # 完整下载（推荐国内镜像）
#   export HF_ENDPOINT=https://hf-mirror.com
#   bash scripts/download_eb_alfred_dataset.sh
#
#   # 仅下载单条任务测试
#   export HF_ENDPOINT=https://hf-mirror.com
#   export EB_DOWNLOAD_MODE=single
#   export EB_TASK_INDEX=0
#   bash scripts/download_eb_alfred_dataset.sh
#
#   # 自动创建符号链接
#   export EB_SYMLINK=1
#   bash scripts/download_eb_alfred_dataset.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBENCH_VENV:=/root/autodl-tmp/conda-envs/embench}"
PY="${EMBENCH_VENV}/bin/python"
"$PY" -m pip install -q huggingface_hub
# hf_transfer 与部分镜像/代理不兼容时，可 export HF_HUB_ENABLE_HF_TRANSFER=0
exec "$PY" "$ROOT/scripts/download_eb_alfred_hf.py" "$@"
