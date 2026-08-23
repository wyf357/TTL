# OpenTTL

Decoder-only 大语言模型**测试时适应**（TTA）研究框架：Hydra 配置 + 可插拔策略（TLM / Tent / EATA）+ Hugging Face **PEFT LoRA** + `transformers.Trainer` + Accelerate。

## 安装

在远程环境（或任意已配置好 CUDA 与 PyTorch 的机器）中：

```bash
cd OpenTTL
pip install -r requirements.txt
# 或：pip install -e .
```

版本下限见 [`requirements.txt`](requirements.txt)（面向 Qwen3.5 等新架构）。`flash-attn` 需与当前 CUDA / PyTorch 匹配的预编译包或本地编译环境；若安装失败，可在 Hydra 的 `model` 配置里将 `attn_implementation` 设为 `sdpa` 作为回退。

## 离线适应

```bash
python scripts/run_offline.py model=qwen35 strategy=tlm data=dummy_tiny train=default
# 首次运行会从 Hub 拉取模型，需网络与足够显存/内存
```

多卡示例：

```bash
accelerate launch scripts/run_offline.py model=llama3 strategy=tent data=dummy_tiny
```

## 在线模拟

```bash
python scripts/run_online.py model=qwen35 strategy=tent data=dummy_tiny train=default
```

## 配置说明

- `configs/model/`：checkpoint、`torch_dtype`、`attn_implementation`、`trust_remote_code`、PEFT 与推荐 `target_modules`。
- `configs/strategy/`：各策略超参（TLM 的 PPL 阈值、Tent 的熵项等）。
- `configs/data/`：HF Dataset 路径、文本列名、最大长度；`dummy_tiny` 用于快速冒烟。

将 `data/adapteval_subset.yaml` 中的占位 repo 与列名换成你的 AdaptEval 数据源即可。

## 评测占位

```bash
python evaluations/run_adapteval.py
```

（需按实际基准补全指标逻辑。）

## BFCL（Berkeley Function Calling Leaderboard）评测

预训练模型在 Gorilla BFCL v3 上做函数调用预测与 AST 评分（近似官方 AST eval；exec_*/multi_turn 仅生成预测）。

```bash
# 可选：预下载数据到本地（离线环境）
python scripts/download_bfcl_dataset.py --out /root/autodl-tmp/bfcl

# 评测（默认 category=simple，模型用 configs/model/default.yaml，可用 model=qwen35 覆盖）
python evaluations/run_bfcl.py category=multiple bfcl_local_root=/root/autodl-tmp/bfcl

# 解析/评分逻辑冒烟测试（无需 GPU/依赖）
python test_bfcl_eval.py
```

产物：`outputs/bfcl_metrics.json`（指标）与 `outputs/bfcl_result.jsonl`（逐样本预测，兼容官方 answer 格式）。
