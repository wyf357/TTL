#!/usr/bin/env bash
# 无物理显示器且 nvidia Xorg 不可用时：用 Xvfb 提供 GLX 显示，再跑 EB-ALFRED 桥接。
# 依赖：apt install xvfb x11-utils（或等价包）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${XVFB_DISPLAY_NUM:=99}"
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

export EB_DISPLAY="$DISP"
exec bash "$ROOT/scripts/run_embodiedbench_autodl.sh" "$@"
