#!/usr/bin/env bash
# 无显示器环境运行 EmbodiedBench 评测（使用 Xvfb 提供 GLX 显示）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBODIEDBENCH_SRC:="$ROOT/third_party/EmbodiedBench"}"
: "${EMBENCH_VENV:=""}"
: "${XVFB_DISPLAY_NUM:=99}"
: "${TTA_GPU:=0}"
: "${RENDER_GPU:=1}"

DISP=":${XVFB_DISPLAY_NUM}"

echo "清理 Xvfb 残留进程和锁文件..." >&2
pkill -9 -f "Xvfb $DISP" 2>/dev/null || true
sleep 0.5

# 删除所有相关锁文件
rm -f /tmp/.X${XVFB_DISPLAY_NUM}-lock
rm -f /tmp/.X11-unix/X${XVFB_DISPLAY_NUM}
sleep 0.5

# ============================================

if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "$DISP" >/dev/null 2>&1; then
  echo "使用已有显示 $DISP" >&2
else
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "错误: 未找到 Xvfb。请安装: apt-get install -y xvfb" >&2
    exit 1
  fi
  echo "启动 Xvfb $DISP（日志 /tmp/openttl_xvfb_${XVFB_DISPLAY_NUM}.log）…" >&2
  
  # 修改点2: 提高分辨率，添加 -ac +bs 参数
  nohup Xvfb "$DISP" -screen 0 1280x720x24 +extension GLX +extension RENDER -ac +bs \
    >>"/tmp/openttl_xvfb_${XVFB_DISPLAY_NUM}.log" 2>&1 &
  
  # 修改点3: 增加等待时间
  for i in $(seq 1 60); do
    sleep 0.5
    if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "$DISP" >/dev/null 2>&1; then
      echo "Xvfb $DISP 启动成功 (尝试 $i 次)" >&2
      break
    fi
    if [ $i -eq 60 ]; then
      echo "错误: Xvfb 启动超时" >&2
      cat "/tmp/openttl_xvfb_${XVFB_DISPLAY_NUM}.log" >&2
      exit 1
    fi
  done
fi

export DISPLAY="$DISP"
export EB_ALF_X_DISPLAY="${XVFB_DISPLAY_NUM}"

# GPU 设置
export CUDA_VISIBLE_DEVICES="$TTA_GPU"
export TTA_CUDA_DEVICE="$TTA_GPU"
export RENDER_CUDA_DEVICE="$RENDER_GPU"
export AI2THOR_USE_GPU=true

export PYTHONPATH="${EMBODIEDBENCH_SRC}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# CUDA-Vulkan 映射
_map="${HOME}/.ai2thor/cuda-vulkan-mapping.json"
if [ ! -f "$_map" ]; then
  mkdir -p "${HOME}/.ai2thor"
  printf '%s\n' "{\"${TTA_GPU}\": ${TTA_GPU}, \"${RENDER_GPU}\": ${RENDER_GPU}}" > "$_map"
  echo "已写入 CUDA↔Vulkan 映射: ${_map}" >&2
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