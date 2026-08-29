#!/usr/bin/env bash
# BFCL simple：Qwen3.5-2B Online TTA（TENT + TLM），nohup 后台顺序跑。
#
# Usage:
#   bash scripts/run_bfcl_simple_tta_nohup.sh
#   BFCL_GPU=1 bash scripts/run_bfcl_simple_tta_nohup.sh
#   bash scripts/run_bfcl_simple_tta_nohup.sh --foreground
#   STRATEGIES=tlm EXTRA_HYDRA='strategy.use_threshold=false' \
#     OUT_DIR=outputs/bfcl_tta_simple_tlm_nothresh \
#     bash scripts/run_bfcl_simple_tta_nohup.sh
#
# 日志与产物目录（固定、清晰）:
#   OpenTTL/outputs/bfcl_tta_simple/
#     run.log              # 总日志
#     tent.log / tlm.log   # 各策略完整 stdout
#     tent_metrics.json / tlm_metrics.json
#     tent_result.jsonl / tlm_result.jsonl
#     run.pid
#
# 查看进度:
#   tail -f outputs/bfcl_tta_simple/run.log
#   tail -f outputs/bfcl_tta_simple/tent.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TTL_ROOT="$(cd "$ROOT/.." && pwd)"

: "${BFCL_GPU:=0}"
: "${BFCL_DATA:="$TTL_ROOT/data/bfcl"}"
: "${BFCL_MODEL_PATH:="$TTL_ROOT/Qwen3.5-2B"}"
: "${PY_TT:=/home/jxy/miniconda3/envs/openttl/bin/python}"
: "${STRATEGIES:=tent tlm}"
: "${EXTRA_HYDRA:=}"
: "${OUT_DIR:=$ROOT/outputs/bfcl_tta_simple}"
PID_FILE="$OUT_DIR/run.pid"
MAIN_LOG="$OUT_DIR/run.log"

export CUDA_VISIBLE_DEVICES="$BFCL_GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -x "$PY_TT" ]; then
    PY_TT="$(command -v python3)"
fi

if [ ! -d "$BFCL_MODEL_PATH" ] || ! compgen -G "$BFCL_MODEL_PATH/*.safetensors" > /dev/null; then
    echo "ERROR: 未找到模型权重: $BFCL_MODEL_PATH" >&2
    exit 1
fi

if [ ! -f "$BFCL_DATA/BFCL_v3_simple.json" ]; then
    echo "ERROR: 未找到 BFCL simple 数据: $BFCL_DATA/BFCL_v3_simple.json" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

run_one() {
    local strat="$1"
    local strat_log="$OUT_DIR/${strat}.log"
    local metrics="$OUT_DIR/${strat}_metrics.json"
    local result="$OUT_DIR/${strat}_result.jsonl"

    {
        echo "========================================"
        echo "[$(date '+%F %T')] START strategy=${strat} category=simple"
        echo "  model=$BFCL_MODEL_PATH"
        echo "  data=$BFCL_DATA"
        echo "  gpu=$BFCL_GPU python=$PY_TT"
        echo "  extra_hydra=${EXTRA_HYDRA:-<none>}"
        echo "  metrics=$metrics"
        echo "  result=$result"
        echo "========================================"
    } | tee -a "$MAIN_LOG"

    set +e
    # EXTRA_HYDRA 故意不引号：允许传入多个 Hydra override
    # shellcheck disable=SC2086
    "$PY_TT" evaluations/run_bfcl.py --config-name=eval_bfcl_online \
        model=qwen35_2b \
        "model.pretrained_model_name_or_path=$BFCL_MODEL_PATH" \
        bfcl_local_root="$BFCL_DATA" \
        category=simple \
        "strategy=$strat" \
        "output_json=$metrics" \
        "result_jsonl=$result" \
        hydra.run.dir="$OUT_DIR/hydra_${strat}" \
        hydra.output_subdir=null \
        $EXTRA_HYDRA \
        >"$strat_log" 2>&1
    local rc=$?
    set -e

    {
        echo "[$(date '+%F %T')] END strategy=${strat} exit=$rc"
        if [ -f "$metrics" ]; then
            echo "[$(date '+%F %T')] metrics:"
            cat "$metrics"
        else
            echo "[$(date '+%F %T')] WARN: missing metrics file $metrics"
            echo "---- last 40 lines of $strat_log ----"
            tail -n 40 "$strat_log" || true
        fi
        echo
    } | tee -a "$MAIN_LOG"

    return "$rc"
}

run_all() {
    cd "$ROOT"
    echo "BFCL simple Online TTA started: $(date '+%F %T')" | tee "$MAIN_LOG"
    echo "OUT_DIR=$OUT_DIR" | tee -a "$MAIN_LOG"
    echo "STRATEGIES=$STRATEGIES" | tee -a "$MAIN_LOG"
    echo "EXTRA_HYDRA=${EXTRA_HYDRA:-<none>}" | tee -a "$MAIN_LOG"
    echo | tee -a "$MAIN_LOG"

    local failed=0
    local s
    for s in $STRATEGIES; do
        if ! run_one "$s"; then
            failed=1
        fi
    done

    echo "BFCL simple Online TTA finished: $(date '+%F %T') failed=$failed" | tee -a "$MAIN_LOG"
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
echo "已用 nohup 启动 BFCL simple Online TTA (STRATEGIES=$STRATEGIES)"
echo "  PID:     $(cat "$PID_FILE")"
echo "  总日志:  $MAIN_LOG"
echo "  分策略:  $OUT_DIR/*.log"
echo "  指标:    $OUT_DIR/*_metrics.json"
echo "查看:      tail -f $MAIN_LOG"
