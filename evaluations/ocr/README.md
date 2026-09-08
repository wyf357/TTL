# OCR 基准评测说明（OCRBench / OmniDocBench，Qwen3.5-2B）

## 目录结构

```
~/ocr_results/
├── ocrbench/Qwen3.5-2B/        # lmms-eval 基线（无 TTA）
├── omnidocbench/Qwen3.5-2B/    # lmms-eval 基线（无 TTA）
├── run_both.log                # 基线运行日志
└── ttl_qwen35_2b/              # TTL（online TTA）评测，OpenTTL 框架
    ├── run_all.log
    ├── ocrbench/               # baseline / tent / tlm / come 各 .json + .jsonl
    └── omnidocbench/           # 同上
```

## 基线结果（lmms-eval，无 TTA）

| 基准 | 分数 |
|---|---|
| OCRBench v1（1000 题） | 84.1% |
| OmniDocBench（981 页） | overall 46.06（text_edit 0.591↓ / table_teds 0.610↑ / formula_edit 0.637↓） |

## TTL 结果（lr=1e-5，bf16，LoRA q/v，流式累积更新）

OCRBench（1000 题，判分与 lmms-eval 同函数）：

| 配置 | 准确率 | 与 baseline 差 |
|---|---|---|
| baseline | 85.60% | — |
| tent | 77.40% | -8.2pp（后段长文抽取退化） |
| tlm | **85.90%** | +0.3pp |
| come | 85.60% | 0.0pp |

OmniDocBench（981 页）：

| 配置 | overall ↑ | text_edit ↓ | table_teds ↑ | formula_edit ↓ |
|---|---|---|---|---|
| baseline | 42.11 | 0.609 | 0.617 | 0.744 |
| tent | 22.72 | 0.747 | 0.281 | 0.853 |（渐进熵崩塌，-19.4pp） |
| tlm | 43.44 | 0.599 | 0.627 | 0.725 |（+1.32pp，三项全改善） |
| come | **45.14** | 0.626 | 0.651 | 0.670 |（**+3.03pp**，表格/公式大幅改善） |

注：TTL 对比以本框架 baseline 为准；lmms-eval 基线与本框架 baseline 的差（OCRBench
-1.5pp、OmniDocBench +3.9pp）来自 harness 差异（消息模板/图像分辨率默认值），
与 TTA 无关。

## 跨基准结论（含 MMBench-CN）

| 算法 | MMBench-CN | OCRBench | OmniDocBench |
|---|---|---|---|
| tent | 持平（lr=1e-4 骤崩） | -8.2pp | -19.4pp（崩塌） |
| tlm | -0.37pp | +0.3pp | +1.32pp |
| come | +0.27pp | 0.0pp | **+3.03pp** |

- **tent 不适合在线流式 TTA**：三个基准上三种崩塌形态（骤崩/退化/渐进崩塌）
- **tlm 最稳**，任务越难收益越明显
- **come 上限最高**，代价是 3-25 倍的计算开销（16 步 rollout）

## 复现命令

```bash
# 基线（lmms-eval）
TASKS=ocrbench     MODEL_PATH=/home/jxy/TTL/Qwen3.5-2B bash evaluations/ocr/run_ocr_eval.sh
TASKS=omnidocbench MODEL_PATH=/home/jxy/TTL/Qwen3.5-2B bash evaluations/ocr/run_ocr_eval.sh

# TTL 全套（OpenTTL；ONLY_BENCH / ONLY / LR / MAX_SAMPLES 可调）
nohup bash OpenTTL/scripts/run_ocr_ttl_strategies.sh > ~/ocr_results/ttl_qwen35_2b/run_all.log 2>&1 &

# 单配置
cd OpenTTL && ~/miniconda3/envs/mmbench/bin/python evaluations/run_ocr_ttl.py \
  benchmark=omnidocbench strategy=come online.enabled=true model.peft.enabled=true \
  model.torch_dtype=bfloat16 online.lr=1e-5 online.sync_every_n_updates=0 \
  output_json=~/ocr_results/ttl_qwen35_2b/omnidocbench/come.json
```

依赖：`python-Levenshtein`、`distance`、`apted`、`editdistance`（mmbench 环境，清华源）。
两个任务 yaml 已删除 `token: True`（免 HF 登录），数据走 hf-mirror。
