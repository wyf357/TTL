#!/usr/bin/env bash
# 大模型 TTA 便捷启动脚本。不修改 OpenTTL 内任何文件。
# 已内置 Qwen3.5-2B（model=qwen35_2b + 本机路径 + 开启 LoRA，供 TTA 使用）。
# 需要改模型位置或策略时，可在命令行后追加参数（Hydra 靠后的覆盖前面的），例如:
#   ./run_causal_lm_tta.sh tta_mode=online model.pretrained_model_name_or_path=/其它/路径
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 与 OpenTTL/configs/model/qwen35_2b.yaml 中默认本地路径一致；按需改本行或运行时用覆写
QWEN35_2B_PATH="${QWEN35_2B_PATH:-/root/autodl-tmp/Qwen3.5-2B}"

exec python "$DIR/run_causal_lm_tta.py" \
  model=qwen35_2b \
  "model.pretrained_model_name_or_path=${QWEN35_2B_PATH}" \
  model.peft.enabled=true \
  "$@"
