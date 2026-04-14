#!/usr/bin/env bash
# 从 hf-mirror 全量下载数据集 EmbodiedBench/EB-ALFRED 到 EmbodiedBench 可用路径：
#   <OpenTTL>/third_party/EmbodiedBench  （解析符号链接后）
#   → .../embodiedbench/envs/eb_alfred/data/json_2.1.0/
#
# 数据源（与网页一致）：
#   https://hf-mirror.com/datasets/EmbodiedBench/EB-ALFRED/tree/main
#
# 用法（在 OpenTTL 仓库根执行或任意目录执行本脚本均可）：
#   bash scripts/download_eb_alfred_full_hf_mirror.sh
#
# 可选环境变量（下载逻辑见 scripts/download_eb_alfred_hf.py）：
#   HF_ENDPOINT           默认 https://hf-mirror.com
#   EB_BATCH_SIZE         每批文件数，默认 32
#   EB_MAX_WORKERS        批内并发，默认 4（镜像易 429 可调低）
#   EB_INTER_BATCH_SLEEP  批间休眠秒数，默认 1
#   EB_MAX_ROUNDS         多轮补全上限，默认 50
#   EB_PER_FILE_RETRIES   单文件重试次数，默认 16
#   EB_USE_HF_TRANSFER=1  启用 hf_transfer 加速（脚本会尝试 pip install）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export EB_BATCH_SIZE="${EB_BATCH_SIZE:-32}"
export EB_MAX_WORKERS="${EB_MAX_WORKERS:-4}"
export EB_INTER_BATCH_SLEEP="${EB_INTER_BATCH_SLEEP:-1}"
export EB_MAX_ROUNDS="${EB_MAX_ROUNDS:-50}"
export EB_PER_FILE_RETRIES="${EB_PER_FILE_RETRIES:-16}"

EB_LINK="$ROOT/third_party/EmbodiedBench"
if test ! -e "$EB_LINK"; then
  echo "错误: 不存在 $EB_LINK" >&2
  echo "请先将 EmbodiedBench 克隆到本机，并在此放置符号链接，例如：" >&2
  echo "  ln -sfn /path/to/EmbodiedBench $EB_LINK" >&2
  exit 1
fi

EB_SRC="$(readlink -f "$EB_LINK")"
if test ! -d "$EB_SRC/embodiedbench"; then
  echo "错误: $EB_SRC 下未找到子目录 embodiedbench/（EmbodiedBench 仓库不完整？）" >&2
  exit 1
fi

DEST="$EB_SRC/embodiedbench/envs/eb_alfred/data/json_2.1.0"
mkdir -p "$(dirname "$DEST")"

PY=""
if test -n "${EMBENCH_VENV:-}" && test -x "${EMBENCH_VENV}/bin/python"; then
  PY="${EMBENCH_VENV}/bin/python"
elif test -x "$ROOT/.venv/bin/python"; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi
if test -z "$PY"; then
  echo "错误: 未找到 python，请设置 EMBENCH_VENV 指向含 huggingface_hub 的 venv" >&2
  exit 1
fi

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "EmbodiedBench 根目录: $EB_SRC"
echo "数据将下载到: $DEST"
echo "使用 Python: $PY"

"$PY" -m pip install -q 'huggingface_hub>=0.20'
EXTRA=()
if test "${EB_USE_HF_TRANSFER:-0}" = "1"; then
  "$PY" -m pip install -q hf_transfer || true
  EXTRA+=(--use-hf-transfer)
fi
exec "$PY" "$ROOT/scripts/download_eb_alfred_hf.py" --mode full --dest "$DEST" "${EXTRA[@]}"
