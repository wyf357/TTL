#!/usr/bin/env bash
# Qwen3.5-2B 上跑 MMBench-CN：baseline（无 TTA）+ tent/eata/come/tlm 四种 TTA 策略。
# 单卡 3090 必须串行；用法：
#   nohup bash OpenTTL/scripts/run_mmbench_cn_strategies.sh > ~/mmbench_cn_results/ttl_qwen35_2b/run_all.log 2>&1 &
# 可选环境变量：ONLY="tent eata"（只跑指定策略） MAX_SAMPLES=200（调试用）
set -uo pipefail

OPENTTL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${MM_PY:-$HOME/miniconda3/envs/mmbench/bin/python}"
OUT_ROOT="${OUT_ROOT:-$HOME/mmbench_cn_results/ttl_qwen35_2b}"
mkdir -p "$OUT_ROOT"

# 数据集走镜像（已缓存则直接用缓存）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$OPENTTL_ROOT"

COMMON=(
  "model.peft.enabled=true"
  # bf16：与 lmms-eval 基线一致（checkpoint 原生精度）；fp16 反传会产生 NaN 梯度
  "model.torch_dtype=bfloat16"
  # TTA 学习率：algorithms/*.md 推荐 1e-5~1e-6（1e-4 已实测导致熵崩塌）
  "online.lr=${LR:-1.0e-5}"
  # HF 后端推理与 TTA 共用同一模型对象，无需导出/同步 LoRA 到磁盘
  "online.sync_every_n_updates=0"
  "online.gradient_checkpointing=true"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  COMMON+=("max_samples=${MAX_SAMPLES}")
fi

run_one() {
  local tag="$1"; shift
  echo "===== [$(date)] ${tag} 开始 ====="
  "$PY" evaluations/run_mmbench_cn.py \
    "${COMMON[@]}" "$@" \
    "output_json=${OUT_ROOT}/${tag}.json"
  echo "===== [$(date)] ${tag} 结束，退出码 $? ====="
}

WANT="${ONLY:-baseline tent eata tlm come}"
for tag in $WANT; do
  case "$tag" in
    baseline)
      run_one baseline online.enabled=false model.peft.enabled=false ;;
    tent|eata|tlm|come)
      run_one "$tag" "strategy=$tag" online.enabled=true ;;
    *)
      echo "未知策略: $tag"; exit 1 ;;
  esac
done
echo "===== [$(date)] 全部完成 ====="
