#!/usr/bin/env bash
# 在数据盘创建 Python venv、安装 PyTorch + LMDeploy + EmbodiedBench 运行 EB-ALFRED（model_type=local）所需依赖。
# 用法：
#   export EMBENCH_VENV=/path/to/embench-venv   # 默认 /root/autodl-tmp/conda-envs/embench
#   export EMBODIEDBENCH_SRC=/path/to/EmbodiedBench  # 默认与仓库 third_party 符号链接目标一致
#   bash scripts/setup_embodiedbench_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBENCH_VENV:=/root/autodl-tmp/conda-envs/embench}"
: "${EMBODIEDBENCH_SRC:="$ROOT/third_party/EmbodiedBench"}"

if test ! -d "$EMBODIEDBENCH_SRC/embodiedbench"; then
  echo "错误: 未找到 EmbodiedBench 源码目录: $EMBODIEDBENCH_SRC（需含子目录 embodiedbench/）" >&2
  exit 1
fi

mkdir -p "$(dirname "$EMBENCH_VENV")"
if test ! -x "$EMBENCH_VENV/bin/python"; then
  PY_BOOT="$(command -v python3 || true)"
  if test -z "$PY_BOOT"; then
    echo "错误: 未找到 python3，无法创建 venv" >&2
    exit 1
  fi
  "$PY_BOOT" -m venv "$EMBENCH_VENV"
fi
PY="$EMBENCH_VENV/bin/python"
PIP="$EMBENCH_VENV/bin/pip"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/pip-tmp}"
mkdir -p "$TMPDIR"

"$PIP" install -U pip setuptools wheel -q
"$PIP" install --no-cache-dir \
  'torch==2.4.0' 'torchvision==0.19.0' \
  --index-url https://download.pytorch.org/whl/cu121
"$PIP" install --no-cache-dir 'protobuf>=4.25.3,<5' 'lmdeploy==0.6.3'
"$PIP" install --no-cache-dir \
  'ai2thor==2.1.0' 'gym==0.23.0' 'hydra-core>=1.3' 'omegaconf>=2.3' \
  'scipy>=1.10,<2' \
  'opencv-python==4.10.0.84' 'h5py' 'vocab' 'revtok' \
  'anthropic' 'google-generativeai' 'google-api-python-client' \
  'qwen-vl-utils' 'grpcio==1.60.0' 'grpcio-status==1.60.0'
"$PIP" install --no-cache-dir 'numpy>=1.23,<2'
# 上游无可靠 PEP660 mapping 时不要用 pip -e；运行脚本用 PYTHONPATH 指向源码根即可
"$PIP" uninstall -y embodiedbench 2>/dev/null || true
rm -f "$EMBENCH_VENV"/lib/python*/site-packages/__editable__.embodiedbench-*.pth 2>/dev/null || true
rm -f "$EMBENCH_VENV"/lib/python*/site-packages/__editable___embodiedbench_*_finder.py 2>/dev/null || true

echo "完成。请运行: bash scripts/download_eb_alfred_dataset.sh"
echo "自检: bash scripts/verify_embodiedbench_readiness.sh"
echo "评测: bash scripts/run_embodiedbench_embench.sh"
