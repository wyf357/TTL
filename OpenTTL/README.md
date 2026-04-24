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

## SGLang 推理 + Online TTA（评测脚本）

依赖：`pip install -e ".[sglang]"` 或 `requirements.txt` 中的 `sglang[all]`。

- **ERQA**：`python evaluations/run_erqa.py`（默认 `inference=sglang`；`online.enabled=false` 仅推理；开启 TTA 时加 `online.enabled=true` 且需 `model.peft.enabled=true`）。
- **MMLU**：`python evaluations/run_mmlu.py inference.backend=sglang`（可选 `online.enabled=true`）。
- **EmbodiedBench**：配置 `tta.enabled=true`、`tta.backend=instruction_entropy`，并与 `model_name` 使用同一 checkpoint；默认走 SGLang 本地后端。若需旧版 transformers 管线，设置环境变量 `OPENTTL_LOCAL_BACKEND=transformers`。

单卡双进程（HF LoRA 训练 + SGLang 推理）时请在 `configs/inference/sglang.yaml` 中调低 `mem_fraction_static`。
