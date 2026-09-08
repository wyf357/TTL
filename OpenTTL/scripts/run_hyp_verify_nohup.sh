#!/usr/bin/env bash
# 假设验证实验（Exp-0/1/2）nohup 后台顺序执行。
#
# Usage:
#   bash scripts/run_hyp_verify_nohup.sh              # nohup 后台
#   bash scripts/run_hyp_verify_nohup.sh --foreground   # 前台
#   RUNS="causal" bash scripts/run_hyp_verify_nohup.sh  # 只跑 Exp-2
#   FORCE=1 bash scripts/run_hyp_verify_nohup.sh        # 忽略已有结果，全部重跑
#
# 查看进度:
#   tail -f outputs/mmstar_hyp/run.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TTL_ROOT="$(cd "$ROOT/.." && pwd)"

: "${MMSTAR_GPU:=0}"
: "${MMSTAR_DATA:=$TTL_ROOT/data/mmstar}"
: "${MMSTAR_MODEL_PATH:=$TTL_ROOT/Qwen3.5-2B}"
: "${PY_TT:=/home/jxy/miniconda3/envs/openttl/bin/python}"
: "${RUNS:=channel observe causal}"
: "${CHANNEL_N:=8}"
: "${OBSERVE_N:=400}"
: "${CAUSAL_N:=200}"
: "${CAUSAL_ORACLE:=1}"
: "${SKIP_DONE:=1}"
: "${FORCE:=0}"
: "${OUT_DIR:=$ROOT/outputs/mmstar_hyp}"

PID_FILE="$OUT_DIR/run.pid"
MAIN_LOG="$OUT_DIR/run.log"

export CUDA_VISIBLE_DEVICES="$MMSTAR_GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

if [ ! -x "$PY_TT" ]; then
    PY_TT="$(command -v python3)"
fi

mkdir -p "$OUT_DIR"

should_skip() {
    local marker="$1"
    if [ "$FORCE" = "1" ]; then
        return 1
    fi
    if [ "$SKIP_DONE" = "1" ] && [ -f "$marker" ]; then
        return 0
    fi
    return 1
}

run_channel() {
    local log="$OUT_DIR/channel.log"
    local marker="$OUT_DIR/channel_results.json"
    if should_skip "$marker"; then
        echo "[$(date '+%F %T')] SKIP channel (exists: $marker)" | tee -a "$MAIN_LOG"
        return 0
    fi
    echo "[$(date '+%F %T')] START Exp-0 channel N=$CHANNEL_N" | tee -a "$MAIN_LOG"
    set +e
    "$PY_TT" "$ROOT/verify_hyp_channel.py" "$CHANNEL_N" >"$log" 2>&1
    local rc=$?
    set -e
    echo "[$(date '+%F %T')] END Exp-0 channel exit=$rc" | tee -a "$MAIN_LOG"
    [ "$rc" -eq 0 ] || return "$rc"
}

run_observe() {
    local log="$OUT_DIR/observe.log"
    local marker="$OUT_DIR/observe_summary.json"
    if should_skip "$marker"; then
        echo "[$(date '+%F %T')] SKIP observe (exists: $marker)" | tee -a "$MAIN_LOG"
        return 0
    fi
    echo "[$(date '+%F %T')] START Exp-1 observe N=$OBSERVE_N" | tee -a "$MAIN_LOG"
    set +e
    "$PY_TT" "$ROOT/verify_hyp_observe.py" "$OBSERVE_N" >"$log" 2>&1
    local rc=$?
    set -e
    echo "[$(date '+%F %T')] END Exp-1 observe exit=$rc" | tee -a "$MAIN_LOG"
    [ "$rc" -eq 0 ] || return "$rc"
}

run_causal() {
    local log="$OUT_DIR/causal.log"
    local marker="$OUT_DIR/causal_summary.json"
    if should_skip "$marker"; then
        echo "[$(date '+%F %T')] SKIP causal (exists: $marker)" | tee -a "$MAIN_LOG"
        return 0
    fi
    local oracle_args=()
    if [ "$CAUSAL_ORACLE" = "1" ]; then
        oracle_args=(--oracle)
    fi
    echo "[$(date '+%F %T')] START Exp-2 causal N=$CAUSAL_N oracle=$CAUSAL_ORACLE" | tee -a "$MAIN_LOG"
    set +e
    "$PY_TT" "$ROOT/verify_hyp_causal.py" "$CAUSAL_N" "${oracle_args[@]}" >"$log" 2>&1
    local rc=$?
    set -e
    echo "[$(date '+%F %T')] END Exp-2 causal exit=$rc" | tee -a "$MAIN_LOG"
    if [ -f "$OUT_DIR/causal_summary.json" ]; then
        echo "[$(date '+%F %T')] verdict:" | tee -a "$MAIN_LOG"
        "$PY_TT" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' \
            "$OUT_DIR/causal_summary.json" | tee -a "$MAIN_LOG"
    fi
    [ "$rc" -eq 0 ] || return "$rc"
}

run_all() {
    cd "$ROOT"
    echo "Hypothesis verify started: $(date '+%F %T')" | tee "$MAIN_LOG"
    echo "OUT_DIR=$OUT_DIR RUNS=$RUNS FORCE=$FORCE SKIP_DONE=$SKIP_DONE" | tee -a "$MAIN_LOG"
    echo "gpu=$MMSTAR_GPU python=$PY_TT" | tee -a "$MAIN_LOG"
    echo | tee -a "$MAIN_LOG"

    local failed=0
    local s
    for s in $RUNS; do
        case "$s" in
            channel) run_channel || failed=1 ;;
            observe) run_observe || failed=1 ;;
            causal)  run_causal  || failed=1 ;;
            *) echo "Unknown run: $s" | tee -a "$MAIN_LOG"; failed=1 ;;
        esac
    done

    echo "Hypothesis verify finished: $(date '+%F %T') failed=$failed" | tee -a "$MAIN_LOG"
    return "$failed"
}

FOREGROUND=0
if [ "${1:-}" = "--foreground" ]; then
    FOREGROUND=1
fi

if [ "$FOREGROUND" = "1" ]; then
    run_all
    exit $?
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "已有任务在运行 (PID=$(cat "$PID_FILE"))"
    echo "日志: $MAIN_LOG"
    exit 1
fi

nohup bash "$0" --foreground >"$OUT_DIR/nohup_wrapper.log" 2>&1 &
echo $! >"$PID_FILE"
echo "已用 nohup 启动假设验证 (RUNS=$RUNS)"
echo "  PID:     $(cat "$PID_FILE")"
echo "  总日志:  $MAIN_LOG"
echo "  分实验:  $OUT_DIR/{channel,observe,causal}.log"
echo "查看:      tail -f $MAIN_LOG"
