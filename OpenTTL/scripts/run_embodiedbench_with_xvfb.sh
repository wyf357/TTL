#!/usr/bin/env bash
# 无显示器环境运行 EmbodiedBench 评测（使用 Xvfb 提供 GLX 显示）
#
# 用法：
#   bash scripts/run_embodiedbench_with_xvfb.sh
#   bash scripts/run_embodiedbench_with_xvfb.sh eb_env=eb-alf n_shots=10 exp_name=my_eval
#   bash scripts/run_embodiedbench_with_xvfb.sh --config-name=eval_embodiedbench_qwen35_9b
#   bash scripts/run_embodiedbench_with_xvfb.sh model_name=/root/autodl-tmp/Qwen3.5-VL-9B-Instruct model_type=local
#
# 可选环境变量：
#   TTA_GPU=0          # TTA/推理使用的 GPU（默认 0）
#   RENDER_GPU=1       # 渲染使用的 GPU（默认 1）
#   XVFB_DISPLAY_NUM=99  # Xvfb 显示号（默认 99）
#
# AI2-THOR（CloudRendering）在无头/Xvfb 下常无法成功运行 vulkaninfo，会误报「请安装 vulkaninfo」。
# 若 ~/.ai2thor/cuda-vulkan-mapping.json 不存在，本脚本会写入默认双卡映射以跳过该探测。
#
# 前置准备：
#   1) bash scripts/setup_embodiedbench_env.sh
#   2) bash scripts/download_eb_alfred_dataset.sh
#   3) bash scripts/verify_embodiedbench_readiness.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBODIEDBENCH_SRC:="$ROOT/third_party/EmbodiedBench"}"
: "${EMBENCH_VENV:=""}"
: "${XVFB_DISPLAY_NUM:=99}"
: "${TTA_GPU:=0}"
: "${RENDER_GPU:=1}"

DISP=":${XVFB_DISPLAY_NUM}"

if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "$DISP" >/dev/null 2>&1; then
  echo "使用已有显示 $DISP" >&2
else
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "错误: 未找到 Xvfb。请安装: apt-get install -y xvfb" >&2
    exit 1
  fi
  echo "启动 Xvfb $DISP（日志 /tmp/openttl_xvfb_${XVFB_DISPLAY_NUM}.log）…" >&2
  nohup Xvfb "$DISP" -screen 0 1024x768x24 +extension GLX +extension RENDER \
    >>"/tmp/openttl_xvfb_${XVFB_DISPLAY_NUM}.log" 2>&1 &
  for _ in $(seq 1 30); do
    sleep 0.3
    if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "$DISP" >/dev/null 2>&1; then
      break
    fi
  done
fi
if command -v xdpyinfo >/dev/null 2>&1 && ! xdpyinfo -display "$DISP" >/dev/null 2>&1; then
  echo "错误: $DISP 仍不可用，请检查 /tmp/openttl_xvfb_${XVFB_DISPLAY_NUM}.log" >&2
  exit 1
fi

export DISPLAY="$DISP"
export EB_ALF_X_DISPLAY="${XVFB_DISPLAY_NUM}"

# GPU 设备隔离：TTA/推理与渲染使用不同 GPU
export CUDA_VISIBLE_DEVICES="$TTA_GPU"
export TTA_CUDA_DEVICE="$TTA_GPU"
export RENDER_CUDA_DEVICE="$RENDER_GPU"

export PYTHONPATH="${EMBODIEDBENCH_SRC}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

_map="${HOME}/.ai2thor/cuda-vulkan-mapping.json"
if [ ! -f "$_map" ]; then
  mkdir -p "${HOME}/.ai2thor"
  printf '%s\n' '{"0": 0, "1": 1}' > "$_map"
  echo "已写入默认 CUDA↔Vulkan 映射（${_map}），避免 vulkaninfo 在无头环境下失败。" >&2
fi

echo "开始运行 EmbodiedBench 评测..." >&2
echo "显示设备: $DISPLAY" >&2
echo "TTA/推理 GPU: $TTA_GPU" >&2
echo "渲染 GPU: $RENDER_GPU" >&2

if [ -n "$EMBENCH_VENV" ] && [ -f "${EMBENCH_VENV}/bin/python" ]; then
  PYTHON_CMD="${EMBENCH_VENV}/bin/python"
else
  PYTHON_CMD="python"
fi

exec "$PYTHON_CMD" "$ROOT/evaluations/run_embodiedbench.py" "$@"
