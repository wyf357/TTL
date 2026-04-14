#!/usr/bin/env bash
# 自检 EmbodiedBench 桥接运行前置（源码、venv、数据、可选本地模型路径）。
# 用法：
#   bash scripts/verify_embodiedbench_readiness.sh
#   EMB_LOCAL_MODEL_PATH=/path/to/HF-snapshot bash scripts/verify_embodiedbench_readiness.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBODIEDBENCH_SRC:="$ROOT/third_party/EmbodiedBench"}"
: "${EMBENCH_VENV:=/root/autodl-tmp/conda-envs/embench}"
PY="${EMBENCH_VENV}/bin/python"
DATA_JSON="${EMBODIEDBENCH_SRC}/embodiedbench/envs/eb_alfred/data/json_2.1.0"

fail() { echo "错误: $*" >&2; exit 1; }
warn() { echo "警告: $*" >&2; }

test -d "$EMBODIEDBENCH_SRC/embodiedbench" || fail "未找到 EmbodiedBench 包: $EMBODIEDBENCH_SRC/embodiedbench（克隆后设 EMBODIEDBENCH_SRC 或 third_party/EmbodiedBench 符号链接）"
test -x "$PY" || fail "未找到 embench venv: $PY（先运行 scripts/setup_embodiedbench_env.sh 或设置 EMBENCH_VENV）"
test -d "$DATA_JSON" || fail "未找到 EB-ALFRED 数据目录: $DATA_JSON（运行 download 脚本并 symlink）"

if test -n "${EMB_LOCAL_MODEL_PATH:-}"; then
  export MP="${EMB_LOCAL_MODEL_PATH}"
  test -d "$MP" || fail "EMB_LOCAL_MODEL_PATH 不是目录: $MP"
  test -f "$MP/config.json" || fail "EMB_LOCAL_MODEL_PATH 缺少 config.json: $MP"
else
  unset MP || true
fi

export PYTHONPATH="${EMBODIEDBENCH_SRC}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# 单次启动 Python，避免重复 import torch/lmdeploy（冷启动可能数分钟）。
"$PY" - <<'PY' || fail "Python 导入与版本检查失败"
import json
import os

import ai2thor
import embodiedbench  # noqa: F401
import lmdeploy
import openttl.eval.embodiedbench_bridge as eb

print("embodiedbench_pkg:", eb.embodiedbench_package_dir())
print("ai2thor:", getattr(ai2thor, "__version__", "?"))
print("lmdeploy:", lmdeploy.__version__)
mp = os.environ.get("MP")
if mp:
    with open(os.path.join(mp, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    mt = str(cfg.get("model_type", "")).lower()
    arch0 = str((cfg.get("architectures") or [""])[0])
    q35 = "qwen3_5" in mt or "Qwen3_5" in arch0
    print("local_model:", mp)
    print("  model_type:", cfg.get("model_type"), "architectures[0]:", arch0)
    print("  OpenTTL turbomind 快捷路径适用:", q35)
PY

if test -z "${DISPLAY:-}"; then
  warn "DISPLAY 未设置。无头跑 EB-ALFRED 前请按 EmbodiedBench README 启动虚拟显示（如 startx / Xvfb）。"
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q GPU; then
  warn "未检测到可用 NVIDIA GPU。model_type=local 时 LMDeploy 通常需要 CUDA；本脚本仍通过目录与配置检查。"
fi

if test -n "${EMB_LOCAL_MODEL_PATH:-}"; then
  if test "${EMB_SKIP_LMDEPLOY_PIPELINE:-}" != 1; then
    if nvidia-smi -L 2>/dev/null | grep -q GPU; then
      "$PY" - <<'PY' || warn "LMDeploy pipeline() 加载失败（检查显存与 lmdeploy 是否支持该架构）"
import os

from lmdeploy import pipeline

p = os.environ["MP"]
pipeline(p)
print("lmdeploy pipeline(load-only): ok")
PY
    else
      warn "跳过 LMDeploy pipeline 实测（无 GPU）。可设置 EMB_SKIP_LMDEPLOY_PIPELINE=1 消除本提示。"
    fi
  fi
else
  echo "提示: 设置 EMB_LOCAL_MODEL_PATH=/你的/HF模型目录 可额外校验 config.json 与（有 GPU 时）LMDeploy 加载。"
fi

echo "EmbodiedBench 前置检查通过。"
