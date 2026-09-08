# 评测结果汇总（Qwen3.5 系列）

本目录收录已完成的全部评测结果。评测代码见 `evaluations/` 与 `OpenTTL/`。

## 目录结构

```
results/
├── android_world/                    # AndroidWorld 摘要（逐任务成功率）
├── mmbench_cn/
│   ├── Qwen3.5-2B/                   # lmms-eval 提交结果（json + xlsx）
│   ├── Qwen3.5-4B/
│   └── ttl_qwen35_2b/                # TTL 评测：5 配置 metrics(.json) + 逐样本(.jsonl) + 日志
├── ocr/
│   ├── ocrbench/                     # lmms-eval 基线 results.json
│   ├── omnidocbench/
│   └── ttl_qwen35_2b/{ocrbench,omnidocbench}/   # TTL 评测 4 配置 + 日志
└── README.md
```

## 1. AndroidWorld（116 任务，M3A 多模态 agent，贪心解码）

| 模型 | 成功率 |
|---|---|
| Qwen3.5-2B | 6.0%（7/116） |
| Qwen3.5-4B | 34.9%（40.5/116） |

## 2. MMBench-CN dev（4329 题，lmms-eval，static 规则判分）

| 模型 | 总准确率 |
|---|---|
| Qwen3.5-2B | 72.94% |
| Qwen3.5-4B | 79.64% |

细分 category/L2 数据见各模型目录下 `mmbench_cn_dev_results.json`。

## 3. MMBench-CN × TTL（Qwen3.5-2B，OpenTTL 框架，lr=1e-5，bf16）

| 配置 | 准确率 | 与 baseline 差 |
|---|---|---|
| baseline（无 TTA） | 81.80% | — |
| tent @ lr=1e-4 | 38.53% | 熵崩塌（恒定输出 D），归档对照 |
| tent @ lr=1e-5 | 81.68% | -0.12pp |
| tlm | 81.43% | -0.37pp |
| come | **82.07%** | **+0.27pp** |

## 4. OCR 基线（lmms-eval，无 TTA）

| 基准 | 分数 |
|---|---|
| OCRBench v1（1000 题） | 84.1% |
| OmniDocBench（981 页） | overall 46.06（text_edit 0.591↓ / table_teds 0.610↑ / formula_edit 0.637↓） |

## 5. OCRBench × TTL（Qwen3.5-2B，lr=1e-5）

| 配置 | 准确率 | 与 baseline 差 |
|---|---|---|
| baseline | 85.60% | — |
| tent | 77.40% | -8.2pp（长文抽取退化） |
| tlm | **85.90%** | +0.3pp |
| come | 85.60% | 0.0pp |

## 6. OmniDocBench × TTL（Qwen3.5-2B，lr=1e-5）

| 配置 | overall ↑ | text_edit ↓ | table_teds ↑ | formula_edit ↓ |
|---|---|---|---|---|
| baseline | 42.11 | 0.609 | 0.617 | 0.744 |
| tent | 22.72 | 0.747 | 0.281 | 0.853（渐进熵崩塌） |
| tlm | 43.44 | 0.599 | 0.627 | 0.725（+1.32pp） |
| come | **45.14** | 0.626 | 0.651 | 0.670（**+3.03pp**） |

## 跨基准 TTL 结论

| 算法 | MMBench-CN | OCRBench | OmniDocBench |
|---|---|---|---|
| tent | 持平/骤崩 | -8.2pp | -19.4pp（崩塌） |
| tlm | -0.37pp | +0.3pp | +1.32pp |
| come | +0.27pp | 0.0pp | **+3.03pp** |

- tent 不适合在线流式 TTA：三个基准三种崩塌形态（详见 `evaluations/mmbench_cn/README.md`）
- tlm 最稳（困惑度门控），任务越难收益越明显
- come 上限最高，计算开销也最大（16 步 rollout，3-25 倍耗时）

注：TTL 各配置对比以同框架 baseline 为准；lmms-eval 基线与本框架 baseline 的
差异来自 harness（消息模板/图像分辨率），与 TTA 无关。复现命令见
`evaluations/mmbench_cn/README.md` 与 `evaluations/ocr/README.md`。
