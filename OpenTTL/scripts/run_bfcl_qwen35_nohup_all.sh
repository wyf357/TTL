#!/usr/bin/env bash
# 使用 nohup 后台批量跑 BFCL v3 评测（Qwen3.5-2B）。
#
# 已跑过 simple 时会自动跳过（除非 FORCE_RERUN=1）。
#
# Usage:
#   bash scripts/run_bfcl_qwen35_nohup_all.sh              # 启动后台任务
#   bash scripts/run_bfcl_qwen35_nohup_all.sh --foreground # 前台顺序跑（调试用）
#   BFCL_GPU=1 bash scripts/run_bfcl_qwen35_nohup_all.sh
#   SKIP_CATEGORIES="live_multiple" bash scripts/run_bfcl_qwen35_nohup_all.sh
#
# 查看进度:
#   tail -f outputs/bfcl_nohup/bfcl_all.log
#   ls outputs/bfcl_nohup/*_metrics.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TTL_ROOT="$(cd "$ROOT/.." && pwd)"

: "${BFCL_GPU:=0}"
: "${BFCL_DATA:="$TTL_ROOT/data/bfcl"}"
: "${BFCL_MODEL_PATH:="$TTL_ROOT/Qwen3.5-2B"}"
: "${PY_TT:=/home/jxy/miniconda3/envs/openttl/bin/python}"
: "${FORCE_RERUN:=0}"
: "${SKIP_CATEGORIES:=}"

LOG_DIR="$ROOT/outputs/bfcl_nohup"
PID_FILE="$LOG_DIR/bfcl_all.pid"
MAIN_LOG="$LOG_DIR/bfcl_all.log"

# BFCL 可自动评分的类别（与 src/openttl/eval/bfcl_eval.py 一致）
# relevance / live_relevance 在官方 Hub 上暂无独立 json，故不包含。
SCORABLE_CATEGORIES=(
    simple
    multiple
    parallel
    parallel_multiple
    java
    javascript
    live_simple
    live_multiple
    live_parallel
    live_parallel_multiple
    irrelevance
    live_irrelevance
)

# 仅生成预测、内置评测不计分（需要执行环境或多轮交互）
# exec_simple exec_multiple exec_parallel exec_parallel_parallel rest sql
# multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context
# multi_turn_composite chatable

export CUDA_VISIBLE_DEVICES="$BFCL_GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ ! -x "$PY_TT" ]; then
    PY_TT="$(command -v python3)"
fi

if [ ! -d "$BFCL_MODEL_PATH" ] || ! compgen -G "$BFCL_MODEL_PATH/*.safetensors" > /dev/null; then
    echo "ERROR: 未找到模型权重: $BFCL_MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

should_skip_category() {
    local cat="$1"
    if [[ " $SKIP_CATEGORIES " == *" $cat "* ]]; then
        return 0
    fi
    if [ "$FORCE_RERUN" != "1" ] && [ -s "$LOG_DIR/${cat}_metrics.json" ]; then
        return 0
    fi
    return 1
}

download_missing_data() {
    local missing=()
    local cat
    for cat in "${SCORABLE_CATEGORIES[@]}"; do
        if [ ! -f "$BFCL_DATA/BFCL_v3_${cat}.json" ]; then
            missing+=("$cat")
        fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then
        echo "[data] 所有可评分类别数据已就绪: $BFCL_DATA"
        return 0
    fi
    echo "[data] 下载缺失类别: ${missing[*]}"
    for cat in "${missing[@]}"; do
        if ! "$PY_TT" "$ROOT/scripts/download_bfcl_dataset.py" \
            --out "$BFCL_DATA" \
            --categories "$cat"; then
            echo "[warn] 无法下载 $cat，后续将跳过该类别" >&2
        fi
    done
}

has_category_data() {
    [ -f "$BFCL_DATA/BFCL_v3_${1}.json" ]
}

run_all_categories() {
    {
        echo "========================================"
        echo "BFCL batch eval started: $(date '+%F %T')"
        echo "Model:  $BFCL_MODEL_PATH"
        echo "Data:   $BFCL_DATA"
        echo "GPU:    $BFCL_GPU"
        echo "Python: $PY_TT"
        echo "LogDir: $LOG_DIR"
        echo "========================================"

        download_missing_data

        local cat
        for cat in "${SCORABLE_CATEGORIES[@]}"; do
            if should_skip_category "$cat"; then
                echo "[skip] $cat（已完成或在 SKIP_CATEGORIES 中）"
                continue
            fi
            if ! has_category_data "$cat"; then
                echo "[skip] $cat（数据文件不存在）"
                continue
            fi

            local cat_log="$LOG_DIR/${cat}.log"
            local metrics_out="$LOG_DIR/${cat}_metrics.json"
            local result_out="$LOG_DIR/${cat}_result.jsonl"

            echo ""
            echo "----------------------------------------"
            echo "[run] category=$cat  $(date '+%F %T')"
            echo "----------------------------------------"

            cd "$ROOT"
            "$PY_TT" evaluations/run_bfcl.py \
                model=qwen35_2b \
                "model.pretrained_model_name_or_path=$BFCL_MODEL_PATH" \
                "bfcl_local_root=$BFCL_DATA" \
                "category=$cat" \
                "output_json=$metrics_out" \
                "result_jsonl=$result_out" \
                2>&1 | tee "$cat_log"

            if [ -s "$metrics_out" ]; then
                echo "[done] $cat -> $metrics_out"
            else
                echo "[warn] $cat 未生成 metrics 文件" >&2
            fi
        done

        echo ""
        echo "========================================"
        echo "BFCL batch eval finished: $(date '+%F %T')"
        echo "汇总指标:"
        for f in "$LOG_DIR"/*_metrics.json; do
            [ -f "$f" ] || continue
            echo "  - $(basename "$f")"
        done
        echo "========================================"
    }
}

if [ "${1:-}" = "--foreground" ]; then
    run_all_categories 2>&1 | tee -a "$MAIN_LOG"
    exit 0
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "已有 BFCL 批量任务在运行 (PID=$(cat "$PID_FILE"))"
    echo "日志: tail -f $MAIN_LOG"
    exit 0
fi

: >"$MAIN_LOG"
nohup bash "$0" --foreground >/dev/null 2>&1 &
echo $! >"$PID_FILE"

echo "已用 nohup 启动 BFCL 批量评测"
echo "  PID:   $(cat "$PID_FILE")"
echo "  日志:  tail -f $MAIN_LOG"
echo "  指标:  ls $LOG_DIR/*_metrics.json"
echo ""
echo "将依次评测以下类别（自动跳过已完成的 simple 等）:"
for cat in "${SCORABLE_CATEGORIES[@]}"; do
    if should_skip_category "$cat"; then
        echo "  [skip] $cat"
    elif ! has_category_data "$cat"; then
        echo "  [skip] $cat（无数据）"
    else
        echo "  [run]  $cat"
    fi
done
