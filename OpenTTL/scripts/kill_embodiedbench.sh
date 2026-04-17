#!/usr/bin/env bash
# 停止正在运行的 EmbodiedBench / OpenTTL 评测及相关 Unity(AI2THOR) 进程
set -euo pipefail
echo "结束 run_embodiedbench / Hydra 评测进程..." >&2
pkill -f "evaluations/run_embodiedbench.py" 2>/dev/null || true
pkill -f "run_embodiedbench.py" 2>/dev/null || true
echo "结束 AI2THOR Unity 可执行体..." >&2
pkill -f "thor-Linux64" 2>/dev/null || true
# 常见 Unity 子进程名（若仍残留可手动 ps 查看）
pkill -f "AI2-THOR" 2>/dev/null || true
echo "完成。若仍有进程，请执行: ps aux | grep -E 'thor|embodiedbench|ai2thor'" >&2
