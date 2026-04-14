#!/usr/bin/env bash
# 检测 NVIDIA + X 是否满足 EB-ALFRED / Ai2THOR（Linux64 + GLX）的常见前置。
# 用法：
#   bash scripts/verify_nvidia_x_stack.sh
#   DISPLAY=:1 bash scripts/verify_nvidia_x_stack.sh
set -euo pipefail

ok() { echo "[OK] $*"; }
warn() { echo "[!!] $*" >&2; }
fail() { echo "[NO] $*" >&2; RC=1; }

RC=0

echo "=== 1) GPU 可见性 (nvidia-smi) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  _nsmi_list=$(nvidia-smi -L 2>/dev/null || true)
  if [[ -n "${_nsmi_list//[$' \t\n']}" ]]; then
    printf '%s\n' "$_nsmi_list" | sed 's/^/[OK] /'
  else
    fail "nvidia-smi 无 GPU 列表（驱动未加载或容器未映射 GPU）"
  fi
else
  fail "未找到 nvidia-smi"
fi

echo ""
echo "=== 2) 设备节点 (/dev) ==="
for n in /dev/nvidiactl /dev/nvidia-uvm; do
  if [[ -e "$n" ]]; then ok "存在 $n"; else fail "缺少 $n"; fi
done
shopt -s nullglob
_any_gpu=0
for n in /dev/nvidia[0-9]*; do
  ok "存在 $n"
  _any_gpu=1
done
shopt -u nullglob
if [[ "$_any_gpu" -eq 0 ]]; then
  fail "未发现 /dev/nvidia0 等字符设备（常见于未 --gpus 的 Docker）"
fi
if [[ -e /dev/dri/card0 ]]; then
  ok "存在 /dev/dri/card0（内核 DRM，部分 X/GL 路径会用到）"
else
  warn "无 /dev/dri/card0（部分容器正常；纯 nvidia X 仍可能可用）"
fi

echo ""
echo "=== 3) PCI 上的 NVIDIA (lspci) ==="
if command -v lspci >/dev/null 2>&1; then
  if lspci 2>/dev/null | grep -qi nvidia; then
    ok "lspci 可见 NVIDIA 设备"
    lspci 2>/dev/null | grep -i nvidia | head -5 || true
  else
    warn "lspci 未列出 NVIDIA（虚拟机/透传异常时可忽略，以 nvidia-smi 为准）"
  fi
else
  warn "未安装 pciutils（无 lspci）；EmbodiedBench startx.py 需要: apt install pciutils"
fi

echo ""
echo "=== 4) X 显示与 GLX（当前 DISPLAY=${DISPLAY:-未设置}）==="
_disp="${DISPLAY:-:0}"
if command -v xdpyinfo >/dev/null 2>&1; then
  if xdpyinfo -display "$_disp" >/dev/null 2>&1; then
    ok "xdpyinfo 可连接 $_disp"
    # 避免 xdpyinfo|grep 在 set -o pipefail 下因 SIGPIPE 误判失败
    _xdout=$(xdpyinfo -display "$_disp" 2>/dev/null || true)
    if [[ "$_xdout" == *GLX* ]]; then
      ok "扩展列表含 GLX（Ai2THOR Linux64 需要）"
    else
      fail "显示 $_disp 上 xdpyinfo 未检测到 GLX（Xvfb 部分配置可无此行；真 NVIDIA X 通常必须有）"
    fi
    _dim=$(printf '%s\n' "$_xdout" | awk '/dimensions:/ {print $2; exit}' || true)
    if [[ -n "$_dim" ]]; then
      _w="${_dim%x*}"
      _h="${_dim#*x}"
      if [[ "$_w" =~ ^[0-9]+$ && "$_h" =~ ^[0-9]+$ && "$_w" -ge 500 && "$_h" -ge 500 ]]; then
        ok "分辨率 ${_w}x${_h}（EmbodiedBench 常用 500 量级，一般足够）"
      else
        warn "分辨率 ${_w}x${_h} 偏小；若 Thor 报分辨率不足请增大虚拟屏"
      fi
    fi
  else
    fail "xdpyinfo 无法连接 $_disp（X 未启动或 DISPLAY 错误）"
    echo "    典型修复: tmux 里运行 python -m embodiedbench.envs.eb_alfred.scripts.startx <显示号>，或 ai2thor-xorg start <号>" >&2
  fi
else
  warn "未安装 x11-utils（无 xdpyinfo）；请: apt install x11-utils"
  fail "无法自动检测 GLX"
fi

echo ""
echo "=== 5) Xorg 与 nvidia 视频包（可选）==="
if command -v Xorg >/dev/null 2>&1; then ok "Xorg 在 PATH 中"; else warn "未找到 Xorg（xserver-xorg-core 未装？）"; fi
if dpkg -l 'xserver-xorg-video-nvidia-*' 2>/dev/null | grep -q ^ii; then
  ok "已安装 xserver-xorg-video-nvidia-*"
else
  warn "未检测到 xserver-xorg-video-nvidia-*（若用 Xvfb 或纯软件路径可能不需要）"
fi

echo ""
echo "=== 6) Docker / 能力变量（若在容器内）==="
if [[ -f /.dockerenv ]]; then
  ok "检测到 /.dockerenv（在容器内）"
  if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then ok "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"; else warn "未设置 NVIDIA_VISIBLE_DEVICES"; fi
  if [[ -n "${NVIDIA_DRIVER_CAPABILITIES:-}" ]]; then ok "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"; else warn "未设置 NVIDIA_DRIVER_CAPABILITIES（图形类任务可设 graphics,compute,utility）"; fi
else
  ok "非 Docker 根文件系统判定（或未使用 /.dockerenv）"
fi

echo ""
if [[ "$RC" -eq 0 ]]; then
  echo "汇总: 关键项通过。若仍无法起 Thor，请查看 Xorg/ai2thor 日志与 ~/.config/unity3d/.../Player.log"
else
  echo "汇总: 存在失败项（见上文 [NO]）。请先解决驱动/设备/DISPLAY 再跑 EmbodiedBench。" >&2
fi
exit "$RC"
