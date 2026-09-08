"""MMBench-CN (dev split) 数据加载与 prompt 构建。

Prompt 格式与 lmms-eval ``mmbench_cn_dev`` 完全对齐
（``third_party/lmms-eval/lmms_eval/tasks/mmbench/cn_utils.py``）：

- options: ``There are several options:\nA. ...\nB. ...``（跳过缺失 / ``'nan'`` 选项）
- query: ``f"{hint} {question} {options}"`` 若 ``pd.notna(hint)``，否则 ``f"{question} {options}"``
  （注意：数据集中 hint 缺失时是字符串 ``'nan'``，``pd.notna('nan')`` 为 True，
  lmms-eval 会原样拼入 prompt；此处保持一致）
- post_prompt: ``\\n请直接使用所提供的选项字母作为答案回答。``
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

SYS_PROMPT = "There are several options:"
POST_PROMPT = "\n请直接使用所提供的选项字母作为答案回答。"
OPTION_CANDIDATES = ["A", "B", "C", "D", "E"]

DEFAULT_DATASET_PATH = "lmms-lab-encoder/MMBench"
DEFAULT_DATASET_NAME = "cn"
DEFAULT_SPLIT = "dev"


def _is_nan_like(value: Any) -> bool:
    """``pd.isna`` 语义：None 或 float NaN。"""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def create_options_prompt(
    row: Dict[str, Any],
    candidates: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """复刻 lmms-eval ``MMBench_Evaluator.create_options_prompt``。"""
    cands = candidates or OPTION_CANDIDATES
    options: Dict[str, str] = {}
    for cand in cands:
        if cand not in row:
            continue
        item = row[cand]
        if _is_nan_like(item):
            continue
        s = str(item)
        if not s or s == "nan":
            continue
        options[cand] = s
    options = dict(sorted(options.items()))
    prompt = f"{SYS_PROMPT}\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
    return prompt, options


def format_mmbench_cn_query(row: Dict[str, Any]) -> str:
    """复刻 lmms-eval ``mmbench_doc_to_text``（含 post_prompt）。"""
    options_prompt, _ = create_options_prompt(row)
    hint = row.get("hint")
    if not _is_nan_like(hint):  # pd.notna(hint)：字符串 'nan' 也会被拼入，与 lmms-eval 一致
        query = f"{hint} {row['question']} {options_prompt}"
    else:
        query = f"{row['question']} {options_prompt}"
    return f"{query}{POST_PROMPT}"


def load_mmbench_cn_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
) -> Any:
    """加载 MMBench-CN；优先走本地 HF 缓存（需先经 hf-mirror 下载过）。"""
    from datasets import load_dataset

    return load_dataset(dataset_path, dataset_name, split=split)
