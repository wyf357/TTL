# MMMU 多模态标准评测（无 TTA）

使用 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 跑 **`mmmu_val`** / **`mmmu_test`** 等任务，与 OpenTTL 的测试时适应流程无关。

## 1. 数据集

lmms-eval 的 `mmmu_val` 使用 Hub 上的 **`lmms-lab/MMMU`**（见上游 `lmms_eval/tasks/mmmu/mmmu_val.yaml`），与仅下载 `MMMU/MMMU` 的用途不同。

在仓库根目录执行（推荐）：

```bash
python download_mmmu_hf_mirror.py --preset lmms-eval
```

会在数据盘生成例如 `/root/autodl-tmp/lmms-lab-MMMU/` 的 snapshot。评测时仍需让 **`datasets` 至少成功加载过一次** `lmms-lab/MMMU`（会写入 `HF_HOME` / `HF_DATASETS_CACHE`），否则请联网先跑验证脚本：

```bash
export HF_HOME=/root/autodl-tmp/hf
pip install datasets
python evaluations/mmmu/verify_mmmu_for_lmms_eval.py
```

## 2. 安装 lmms-eval

```bash
pip install -r requirements-mmmu-eval.txt
```

或与官方一致从源码 `uv pip install -e ".[all]"`（依赖更全）。

## 3. 单次评测

根据权重架构设置 `--model`（`python -m lmms_eval --help` 或官方 [current_tasks / models](https://github.com/EvolvingLMMs-Lab/lmms-eval)）：

| 常见权重族 | 可尝试的 `LMMS_MODEL` |
|------------|------------------------|
| Qwen3-VL / Qwen3.5 多模态（Transformers 视觉分支） | `qwen3_vl` |
| Qwen2.5-VL | `qwen2_5_vl` |
| Gemma 3 多模态 | `gemma3` |

示例：

```bash
export HF_HOME=/root/autodl-tmp/hf
export TASKS=mmmu_val          # 或 mmmu_test
export LMMS_MODEL=qwen3_vl
export MODEL_PATH=/root/autodl-tmp/Qwen3.5-4B
bash evaluations/mmmu/run_mmmu_baseline.sh
```

可选环境变量：`NUM_PROCESSES`、`MAX_PIXELS`、`ATTN_IMPLEMENTATION`（默认 `sdpa`）、`INTERLEAVE_VISUALS`（MMMU 说明见上游 `examples/models/qwen25vl.sh` 注释）、`LOG_SAMPLES=1`。

## 4. 三台机器上默认路径连跑

```bash
export ROOT_TMP=/root/autodl-tmp
# 若封装名与默认不一致：
# export LMMS_MODEL_QWEN35_2B=qwen3_vl
bash evaluations/mmmu/run_autodl_three_models.sh
```

默认假设模型目录为 `Qwen3.5-2B`、`Qwen3.5-4B`、`gemma_E2B`；若目录名不同，请改 `run_autodl_three_models.sh` 或传入覆盖变量（可自行编辑脚本内 `run_one` 参数）。

## 5. 不要走 OpenTTL TTA

请勿使用 `OpenTTL/scripts/run_offline.py` / `run_online.py` 做本 baseline；那些入口会走 `TTATrainer` 适应逻辑。
