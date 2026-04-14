#!/usr/bin/env bash
# AutoDL / 无物理显示器 + NVIDIA：跑 EB-ALFRED 桥接，默认本地 Qwen3.5-2B。
#
# 前置（一次性）：
#   1) bash scripts/setup_embodiedbench_env.sh
#   2) 数据与 symlink：bash scripts/download_eb_alfred_dataset.sh（或已对齐 json_2.1.0）
#
# X11（必须）：EBAlfEnv 使用显示号 1，请保持 export DISPLAY=:1 与下面 startx 参数一致。
#   在另一个 tmux 窗口执行（会占用前台，勿关）：
#     export PYTHONPATH="$EMBODIEDBENCH_SRC:$OPENTTL_ROOT/src"
#     "$EMBENCH_VENV/bin/python" -m embodiedbench.envs.eb_alfred.scripts.startx 1
#   官方说明：EmbodiedBench README「headless」「startx」。
#
# 一键评测（默认假设 :1 上已有 X）：
#   bash scripts/run_embodiedbench_autodl.sh
#
# 可选环境变量：
#   EB_AUTODL_MODEL   本地模型目录（默认 /root/autodl-tmp/Qwen3.5-2B）
#   EMB_AUTO_START_X=1  若 :1 无响应，则尝试后台 nohup 启动 startx（需 root、可能失败，失败请用手动 tmux 方式）
#   其余参数原样传给 Hydra，例如：selected_indexes='[0]' exp_name=smoke
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EMBODIEDBENCH_SRC:="$ROOT/third_party/EmbodiedBench"}"
: "${EMBENCH_VENV:=/root/autodl-tmp/conda-envs/embench}"
: "${EB_AUTODL_MODEL:=/root/autodl-tmp/Qwen3.5-2B}"
PY="${EMBENCH_VENV}/bin/python"

# 与 EmbodiedBench EBAlfEnv.X_DISPLAY 默认一致；若你改过上游 X_DISPLAY，请设置 EB_DISPLAY 与本脚本一致。
: "${EB_DISPLAY:=:1}"
export DISPLAY="$EB_DISPLAY"
# EBAlfEnv / ThorConnector 的 x_display 与 DISPLAY 显示号一致（如 :99 → 99）
export EB_ALF_X_DISPLAY="${EB_DISPLAY#:}"
export PYTHONPATH="${EMBODIEDBENCH_SRC}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v xdpyinfo >/dev/null 2>&1; then
  echo "提示: 未安装 xdpyinfo（包名通常为 x11-utils），无法检测 :1 是否可用；将直接继续。" >&2
elif ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  if [[ "${EMB_AUTO_START_X:-0}" == "1" ]]; then
    _xnum="${DISPLAY#:}"
    echo "DISPLAY=${DISPLAY} 无响应，尝试 EMB_AUTO_START_X 后台启动 Xorg（显示号 ${_xnum}，日志 /tmp/eb_embodiedbench_xorg.log）…" >&2
    # nohup 子进程 PATH 常过窄，需能找到 lspci（pciutils）与 Xorg（xserver-xorg-core）
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
    export PATH
    nohup "$PY" -m embodiedbench.envs.eb_alfred.scripts.startx "${_xnum}" >>/tmp/eb_embodiedbench_xorg.log 2>&1 &
    for _ in $(seq 1 30); do
      sleep 2
      xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 && break
    done
  fi
fi
if command -v xdpyinfo >/dev/null 2>&1 && ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  cat >&2 <<EOF
错误: 当前 DISPLAY=${DISPLAY} 上没有可用的 X 服务，Ai2THOR 无法启动。

请新开一个终端 / tmux 窗口，在仓库根目录执行（保持运行，不要退出；末尾数字须与 EB_DISPLAY 一致，默认 :1 对应 1）：
  export PYTHONPATH="$EMBODIEDBENCH_SRC:$ROOT/src"
  $PY -m embodiedbench.envs.eb_alfred.scripts.startx 1

然后在本终端执行：
  bash scripts/run_embodiedbench_autodl.sh

或单次尝试自动起 X（需 root、且机器已配置 nvidia/Xorg）：
  EMB_AUTO_START_X=1 bash scripts/run_embodiedbench_autodl.sh
EOF
  exit 1
fi

test -d "$EB_AUTODL_MODEL" || { echo "错误: 模型目录不存在: $EB_AUTODL_MODEL（设置 EB_AUTODL_MODEL）" >&2; exit 1; }
test -f "$EB_AUTODL_MODEL/config.json" || { echo "错误: 缺少 config.json: $EB_AUTODL_MODEL" >&2; exit 1; }

exec bash "$ROOT/scripts/run_embodiedbench_embench.sh" \
  model_name="$EB_AUTODL_MODEL" \
  model_type=local \
  "$@"
