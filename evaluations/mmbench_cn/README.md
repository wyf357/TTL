# MMBench-CN 评测说明（Qwen3.5 系列）

本文档记录 Qwen3.5-2B / Qwen3.5-4B 在 MMBench-CN 基准上的评测方法、结果和复现步骤。

## 评测设置

- **基准**：MMBench-CN `dev` split（`mmbench_cn_dev`），共 4329 道中文单选题
- **评测框架**：[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 0.7.2（源码安装自 `third_party/lmms-eval`）
- **模型封装**：`qwen3_5`（多模态，`Qwen3_5ForConditionalGeneration`）
- **判分方式**：`static` 规则判分（本地补丁，详见「注意事项」）
- **解码**：贪心（`temperature=0`），`max_new_tokens=1024`，`enable_thinking=False`
- **数据集下载**：走 `HF_ENDPOINT=https://hf-mirror.com`（TUNA 不镜像 HF hub）
- **环境**：conda 环境 `mmbench`（Python 3.11，torch 2.6.0+cu124，transformers 5.16.1），单卡 RTX 3090 24GB，batch_size=1 串行推理

## 总体结果

| 模型 | 总准确率 | 答对题数 |
|---|---|---|
| Qwen3.5-2B | **72.94%** | ~3158 / 4329 |
| Qwen3.5-4B | **79.64%** | ~3448 / 4329 |

## L2 大类细分

| L2 类别 | 2B | 4B |
|---|---|---|
| coarse_perception（粗粒度感知） | 85.81% | 86.82% |
| finegrained_perception (instance-level)（细粒度感知-单实例） | 76.11% | 84.98% |
| finegrained_perception (cross-instance)（细粒度感知-跨实例） | 68.53% | 71.33% |
| attribute_reasoning（属性推理） | 73.87% | 78.89% |
| relation_reasoning（关系推理） | 66.09% | 82.61% |
| logic_reasoning（逻辑推理） | 43.22% | 56.78% |

## 细粒度类别（category）细分

| 类别 | 2B | 4B |
|---|---|---|
| image_scene | 95.19% | 98.08% |
| identity_reasoning | 93.33% | 95.56% |
| celebrity_recognition | 90.91% | 94.95% |
| social_relation | 90.70% | 90.70% |
| action_recognition | 90.74% | 92.59% |
| image_topic | 88.89% | 91.67% |
| image_style | 86.79% | 88.68% |
| image_emotion | 84.00% | 82.00% |
| function_reasoning | 83.54% | 82.28% |
| attribute_recognition | 79.73% | 86.49% |
| attribute_comparison | 77.27% | 68.18% |
| ocr | 71.79% | 89.74% |
| object_localization | 56.79% | 69.14% |
| image_quality | 66.04% | 64.15% |
| physical_property_reasoning | 52.00% | 65.33% |
| nature_relation | 52.08% | 85.42% |
| physical_relation | 50.00% | 62.50% |
| future_prediction | 52.50% | 50.00% |
| spatial_relationship | 33.33% | 48.89% |
| structuralized_imagetext_understanding | 38.46% | 60.26% |

**结论**：两个模型在感知类任务（场景、人物、社会关系）上表现很好；弱项集中在空间关系、结构化图文理解和逻辑推理。4B 相比 2B 提升约 6.7 个百分点，主要来自 OCR（+17.7）、nature_relation（+33.3）、结构化图文理解（+21.8）等。

## 结果文件位置

```
~/mmbench_cn_results/
├── Qwen3.5-2B/
│   ├── submissions/
│   │   ├── mmbench_cn_dev_results.json   # 总分 + category/l2_category 细分
│   │   └── mmbench_cn_dev_results.xlsx   # 同内容的表格版
│   └── TTL__Qwen3.5-2B/                  # lmms-eval 原始输出
├── Qwen3.5-4B/
│   └── ...（结构同上）
└── run_both.log                          # 两模型串行运行的完整日志
```

快速查看总分：

```bash
python3 -c "
import json
for tag in ['Qwen3.5-2B','Qwen3.5-4B']:
    d = json.load(open(f'/home/jxy/mmbench_cn_results/{tag}/submissions/mmbench_cn_dev_results.json'))
    print(tag, round(d['overall_acc']*100, 2), '%')"
```

## 如何运行

### 前置条件

1. 本地模型权重：`/home/jxy/TTL/Qwen3.5-2B`、`/home/jxy/TTL/Qwen3.5-4B`
   （如需重下，用项目根目录的 `download_qwen35_2b_modelscope.py` 等脚本，走 ModelScope）
2. conda 环境 `mmbench` 已装好（lmms-eval 0.7.2 + torch 2.6.0+cu124）。
   如需新建：
   ```bash
   conda create -n mmbench python=3.11 -y
   ~/miniconda3/envs/mmbench/bin/pip install torch==2.6.0 \
     -f https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124/ \
     -i https://pypi.tuna.tsinghua.edu.cn/simple
   ~/miniconda3/envs/mmbench/bin/pip install -e third_party/lmms-eval --no-deps \
     -i https://pypi.tuna.tsinghua.edu.cn/simple
   # 再按需补装 transformers、accelerate、qwen-vl-utils 等依赖
   ```

### 单模型评测

```bash
cd /home/jxy/TTL_CLONE

# 2B
MODEL_PATH=/home/jxy/TTL/Qwen3.5-2B bash evaluations/mmbench_cn/run_mmbench_cn.sh

# 4B
MODEL_PATH=/home/jxy/TTL/Qwen3.5-4B bash evaluations/mmbench_cn/run_mmbench_cn.sh
```

首次运行会自动从 hf-mirror 下载数据集（`lmms-lab/MMBench-CN`，约 1GB），之后走本地缓存 `~/.cache/huggingface`。

### 两个模型顺序跑（后台 + nohup）

```bash
nohup bash evaluations/mmbench_cn/run_both.sh > ~/mmbench_cn_results/run_both.log 2>&1 &

# 查看进度
tail -f ~/mmbench_cn_results/run_both.log
```

单卡 24GB 显存下两个模型必须串行，完整跑完约需数小时。

### 常用可配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | （必填） | 本地模型目录 |
| `TASKS` | `mmbench_cn_dev` | 评测任务，可换成 `mmbench_en_dev` 等其他基准 |
| `OUTPUT_PATH` | `~/mmbench_cn_results/<模型名>` | 结果输出目录 |
| `BATCH_SIZE` | `1` | 批大小，3090 上建议保持 1 |
| `MAX_NEW_TOKENS` | `1024` | 生成上限 |
| `MAX_PIXELS` | `12845056` | 图像分辨率上限 |
| `LOG_SAMPLES` | `0` | 设为 `1` 时保存每条样本的输出（调试用） |

示例——换英文版基准并保存样本输出：

```bash
TASKS=mmbench_en_dev LOG_SAMPLES=1 \
MODEL_PATH=/home/jxy/TTL/Qwen3.5-4B bash evaluations/mmbench_cn/run_mmbench_cn.sh
```

## 注意事项

1. **lmms-eval 本地补丁**（`third_party/lmms-eval`，不要 `git checkout` 覆盖）：
   - `_default_template_mmbench_cn_yaml` 删除了 `token: True`（免 HF 登录）
   - `cn_utils.py` 判分方式 `eval_method` 由 `"openai"` 改为 `"static"`（规则判分，无需 LLM judge）
2. torch 必须是 cu124 版本（驱动 535 不支持 cu13x），安装时用清华 pytorch-wheels 镜像，直连 pytorch.org 会卡死。
3. 换其他基准（如 `mmmu_val`）用同一脚本改 `TASKS=` 即可，但若该数据集 yaml 里也有 `token: True` 需要同样删除。

---

# 附：TTL（测试时学习）算法评测（Qwen3.5-2B）

在 MMBench-CN 上用 OpenTTL 框架（`OpenTTL/evaluations/run_mmbench_cn.py`）评测了
tent / tlm / come 三种 online TTA 策略（eata 按用户要求取消）。协议：每题先用当前
权重答题判分，再用该题做一步 TTA 更新（不接触标签），LoRA 权重随样本流式累积。

## 总体结果（4329 题，bf16，lr=1e-5）

| 配置 | 准确率 | 与 baseline 差 |
|---|---|---|
| baseline（无 TTA） | 81.80% (3541/4329) | — |
| tent @ lr=1e-4 | 38.53% (1668/4329) | -43.3pp（熵崩塌，见下） |
| tent @ lr=1e-5 | 81.68% (3536/4329) | -0.12pp |
| tlm @ lr=1e-5 | 81.43% (3525/4329) | -0.37pp |
| come @ lr=1e-5 | **82.07%** (3553/4329) | **+0.27pp** |

- come 是唯一超过 baseline 的策略，增益主要来自 logic_reasoning（63.8→66.0）
  和 structuralized_imagetext_understanding（63.8→67.4）。
- MMBench-CN 与模型训练分布接近，TTA 收益空间本来就小，持平/微增属正常结果。

## 熵崩塌现象（tent @ lr=1e-4）

框架默认 lr=1e-4 时 tent 在约第 950 个样本处崩塌：预测退化为恒定输出 "D"
（准确率≈D 的 base rate ~22-25%），熵降到 ~1e-6 后梯度消失、无法自愈。
`algorithms/tent.md` 明确推荐 lr 1e-5~1e-6；改 1e-5 后全程稳定（tent.json）。
崩塌数据归档于 `tent_lr1e-4.json`。

## 结果文件

```
~/mmbench_cn_results/ttl_qwen35_2b/
├── baseline.json / .jsonl          # 无 TTA 基线
├── tent.json / .jsonl              # tent @ lr=1e-5
├── tent_lr1e-4.json / .jsonl       # tent @ lr=1e-4（崩塌，对照）
├── tlm.json / .jsonl               # tlm @ lr=1e-5
├── come.json / .jsonl              # come @ lr=1e-5
└── run_all.log                     # 全部运行日志
```

`.json` 含总分 + category/L2 细分 + 完整 config；`.jsonl` 为逐样本记录
（预测、gold、tta_loss），支持断点续跑（重跑同 output_json 自动跳过已完成样本）。

## 复现命令

```bash
# 全部（baseline+tent+tlm+come；加 eata 用 ONLY="baseline tent eata tlm come"）
nohup bash OpenTTL/scripts/run_mmbench_cn_strategies.sh > ~/mmbench_cn_results/ttl_qwen35_2b/run_all.log 2>&1 &

# 单策略（如 tent，lr 可覆盖）
cd OpenTTL && ~/miniconda3/envs/mmbench/bin/python evaluations/run_mmbench_cn.py \
  strategy=tent online.enabled=true model.peft.enabled=true \
  model.torch_dtype=bfloat16 online.lr=1e-5 online.sync_every_n_updates=0 \
  output_json=~/mmbench_cn_results/ttl_qwen35_2b/tent.json
```

## 为此修复的 OpenTTL 框架问题（勿回滚）

- `adapters/auto.py`：拼接 response 后同步扩展 `mm_token_type_ids`（否则 get_rope_index 报错）
- `strategies/{tta_shared,eata,come,tlm}.py`：熵/NLL 计算升 fp32（fp16 下 eps 舍入为 0 → NaN）
- `strategies/come.py`：`vocab_size` 从 `text_config` 回退取；rollout 每步传图像张量；
  逐步 backward（梯度累积）解决 16 步图累积 OOM
- `online/tta_runner.py`：`sync_every_n_updates or 1` 改为尊重显式 0；支持策略
  `handles_own_backward`（COME 内部自行反传）
- 注意 `configs/online/tta.yaml`、`configs/train/eval_minimal.yaml` 带 `# @package _global_`，
  组内键会平铺到根命名空间，`eval_mmbench_cn.yaml` 因此内联了 online/train 节
