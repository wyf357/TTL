#!/usr/bin/env bash
# Qwen3.5-2B 上跑 OCR 基准的 TTL 评测：OCRBench + OmniDocBench × (baseline, tent, tlm, come)。
# 用法:
#   nohup bash OpenTTL/scripts/run_ocr_ttl_strategies.sh > ~/ocr_results/ttl_qwen35_2b/run_all.log 2>&1 &
# 可选环境变量:
#   ONLY_BENCH="ocrbench"            只跑某个基准（默认 "ocrbench omnidocbench"）
#   ONLY="baseline tent"             只跑指定配置（默认 "baseline tent tlm come"）
#   LR=1e-5                          TTA 学习率
#   MAX_SAMPLES=50                   调试用
# 目录结构: $OUT_ROOT/<benchmark>/<strategy>.json(.jsonl)
set -uo pipefail

OPENTTL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${MM_PY:-$HOME/miniconda3/envs/mmbench/bin/python}"
OUT_ROOT="${OUT_ROOT:-$HOME/ocr_results/ttl_qwen35_2b}"
mkdir -p "$OUT_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$OPENTTL_ROOT"

COMMON=(
  "model.peft.enabled=true"
  # bf16：与 lmms-eval 基线一致；fp16 反传会产生 NaN 梯度
  "model.torch_dtype=bfloat16"
  # algorithms/*.md 推荐 1e-5~1e-6（1e-4 已实测熵崩塌）
  "online.lr=${LR:-1.0e-5}"
  # HF 后端推理与 TTA 共用同一模型对象，无需导出/同步 LoRA
  "online.sync_every_n_updates=0"
  "online.gradient_checkpointing=true"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  COMMON+=("max_samples=${MAX_SAMPLES}")
fi

run_one() {
  local bench="$1" tag="$2"; shift 2
  mkdir -p "${OUT_ROOT}/${bench}"
  echo "===== [$(date)] ${bench}/${tag} 开始 ====="
  "$PY" evaluations/run_ocr_ttl.py \
    "benchmark=${bench}" "${COMMON[@]}" "$@" \
    "output_json=${OUT_ROOT}/${bench}/${tag}.json"
  echo "===== [$(date)] ${bench}/${tag} 结束，退出码 $? ====="
}

BENCHES="${ONLY_BENCH:-ocrbench omnidocbench}"
TAGS="${ONLY:-baseline tent tlm come}"
for bench in $BENCHES; do
  for tag in $TAGS; do
    case "$tag" in
      baseline)
        run_one "$bench" baseline online.enabled=false model.peft.enabled=false ;;
      tent|eata|tlm|come)
        run_one "$bench" "$tag" "strategy=$tag" online.enabled=true ;;
      *)
        echo "未知策略: $tag"; exit 1 ;;
    esac
  done
done
echo "===== [$(date)] 全部完成 ====="
