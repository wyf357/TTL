#!/usr/bin/env bash
# MMStar：Qwen3.5-2B Online TTA（baseline + TENT + TLM），nohup 后台顺序跑。
#
# Usage:
#   bash scripts/run_mmstar_tta_nohup.sh
#   MMSTAR_GPU=1 bash scripts/run_mmstar_tta_nohup.sh
#   bash scripts/run_mmstar_tta_nohup.sh --foreground
#   RUNS="tent tlm" EXTRA_HYDRA='strategy.use_threshold=false' \
#     OUT_DIR=outputs/mmstar_tta_tlm_nothresh \
#     bash scripts/run_mmstar_tta_nohup.sh
#
# 日志与产物目录（固定、清晰）:
#   OpenTTL/outputs/mmstar_tta/
#     run.log                          # 总日志
#     baseline.log / tent.log / tlm.log   # 各 run 完整 stdout
#     baseline_metrics.json / tent_metrics.json / tlm_metrics.json
#     mmstar_qwen35_2B.csv（逐题 CSV，每个 run 覆盖同名文件前先被重命名保留）
#     run.pid
#
# 查看进度:
#   tail -f outputs/mmstar_tta/run.log
#   tail -f outputs/mmstar_tta/tent.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TTL_ROOT="$(cd "$ROOT/.." && pwd)"

: "${MMSTAR_GPU:=0}"
: "${MMSTAR_DATA:="$TTL_ROOT/data/mmstar"}"
: "${MMSTAR_MODEL_PATH:="$TTL_ROOT/Qwen3.5-2B"}"
: "${PY_TT:=/home/jxy/miniconda3/envs/openttl/bin/python}"
: "${RUNS:=baseline tent tlm}"
: "${EXTRA_HYDRA:=}"
: "${OUT_DIR:=$ROOT/outputs/mmstar_tta}"
PID_FILE="$OUT_DIR/run.pid"
MAIN_LOG="$OUT_DIR/run.log"

export CUDA_VISIBLE_DEVICES="$MMSTAR_GPU"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 重定向到日志文件时禁用 stdout 缓冲，保证 tail -f 能实时看到进度
export PYTHONUNBUFFERED=1

if [ ! -x "$PY_TT" ]; then
    PY_TT="$(command -v python3)"
fi

if [ ! -d "$MMSTAR_MODEL_PATH" ] || ! compgen -G "$MMSTAR_MODEL_PATH/*.safetensors" > /dev/null; then
    echo "ERROR: 未找到模型权重: $MMSTAR_MODEL_PATH" >&2
    exit 1
fi

if ! compgen -G "$MMSTAR_DATA/*.parquet" > /dev/null; then
    echo "WARN: 未找到本地 MMStar 数据 ($MMSTAR_DATA)，将回退到 Hub 下载 (HF_ENDPOINT=$HF_ENDPOINT)" >&2
fi

mkdir -p "$OUT_DIR"

run_one() {
    local name="$1"
    local run_log="$OUT_DIR/${name}.log"
    local metrics="$OUT_DIR/${name}_metrics.json"
    local hydra_dir="$OUT_DIR/hydra_${name}"

    # 逐题 CSV 固定名（按模型），run 结束后会拷贝为带 run 名的副本。
    local csv="$OUT_DIR/mmstar_qwen35_2B.csv"

    local cfg_name="eval_mmstar_online"
    local strat_args=()
    if [ "$name" = "baseline" ]; then
        cfg_name="eval_mmstar"
    else
        strat_args=("strategy=$name")
    fi

    {
        echo "========================================"
        echo "[$(date '+%F %T')] START run=${name}"
        echo "  model=$MMSTAR_MODEL_PATH"
        echo "  data=$MMSTAR_DATA"
        echo "  gpu=$MMSTAR_GPU python=$PY_TT"
        echo "  config=$cfg_name extra_hydra=${EXTRA_HYDRA:-<none>}"
        echo "  metrics=$metrics"
        echo "========================================"
    } | tee -a "$MAIN_LOG"

    set +e
    # EXTRA_HYDRA 故意不引号：允许传入多个 Hydra override
    # shellcheck disable=SC2086
    "$PY_TT" evaluations/run_mmstar.py --config-name="$cfg_name" \
        model=qwen35_2b \
        "model.pretrained_model_name_or_path=$MMSTAR_MODEL_PATH" \
        "mmstar_local_root=$MMSTAR_DATA" \
        "${strat_args[@]}" \
        "output_json=$metrics" \
        hydra.run.dir="$hydra_dir" \
        hydra.output_subdir=null \
        $EXTRA_HYDRA \
        >"$run_log" 2>&1
    local rc=$?
    set -e

    if [ -f "$csv" ]; then
        cp -f "$csv" "$OUT_DIR/${name}_mmstar_qwen35_2B.csv" || true
    fi

    {
        echo "[$(date '+%F %T')] END run=${name} exit=$rc"
        if [ -f "$metrics" ]; then
            echo "[$(date '+%F %T')] metrics (accuracy/correct/total/unextracted):"
            "$PY_TT" -c '
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
acc, cor, tot, une = m["accuracy"], m["correct"], m["total"], m.get("unextracted", 0)
print(f"accuracy={acc:.4f} correct={cor} total={tot} unextracted={une}")
for k, v in sorted(m.get("per_category", {}).items()):
    ca, ta, va = v["correct"], v["total"], v["accuracy"]
    print(f"  [{k}] acc={va:.3f} ({ca}/{ta})")
' "$metrics" | tee -a "$MAIN_LOG"
        else
            echo "[$(date '+%F %T')] WARN: missing metrics file $metrics"
            echo "---- last 40 lines of $run_log ----"
            tail -n 40 "$run_log" || true
        fi
        echo
    } | tee -a "$MAIN_LOG"

    return "$rc"
}

run_all() {
    cd "$ROOT"
    echo "MMStar Online TTA started: $(date '+%F %T')" | tee "$MAIN_LOG"
    echo "OUT_DIR=$OUT_DIR" | tee -a "$MAIN_LOG"
    echo "RUNS=$RUNS" | tee -a "$MAIN_LOG"
    echo "EXTRA_HYDRA=${EXTRA_HYDRA:-<none>}" | tee -a "$MAIN_LOG"
    echo | tee -a "$MAIN_LOG"

    local failed=0
    local s
    for s in $RUNS; do
        if ! run_one "$s"; then
            failed=1
        fi
    done

    echo "MMStar Online TTA finished: $(date '+%F %T') failed=$failed" | tee -a "$MAIN_LOG"
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
echo "已用 nohup 启动 MMStar Online TTA (RUNS=$RUNS)"
echo "  PID:     $(cat "$PID_FILE")"
echo "  总日志:  $MAIN_LOG"
echo "  分 run:  $OUT_DIR/*.log"
echo "  指标:    $OUT_DIR/*_metrics.json"
echo "查看:      tail -f $MAIN_LOG"
