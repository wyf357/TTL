"""BFCL (Berkeley Function Calling Leaderboard) 数据加载与提示构造。

数据来源：gorilla-llm/Berkeley-Function-Calling-Leaderboard（BFCL v3）。
优先级：bfcl_local_root（本地 BFCL_v3_*.json）> Hub（需网络）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# 官方 BFCL v3 类别（AST 可评测 / 需执行环境 / 交互式多轮）
BFCL_AST_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)
BFCL_IRRELEVANCE_CATEGORIES = ("irrelevance", "live_irrelevance")
BFCL_RELEVANCE_CATEGORIES = ("relevance", "live_relevance")
BFCL_EXEC_CATEGORIES = (
    "exec_simple",
    "exec_multiple",
    "exec_parallel",
    "exec_parallel_multiple",
    "rest",
    "sql",
)
BFCL_MULTI_TURN_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "multi_turn_composite",
    "chatable",
)
BFCL_ALL_CATEGORIES = (
    BFCL_AST_CATEGORIES
    + BFCL_IRRELEVANCE_CATEGORIES
    + BFCL_RELEVANCE_CATEGORIES
    + BFCL_EXEC_CATEGORIES
    + BFCL_MULTI_TURN_CATEGORIES
)

BFCL_HF_PATH = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# 官方 BFCL system prompt（gorilla 仓库 handler 使用的措辞）
BFCL_SYSTEM_MESSAGE = (
    "You are an expert in composing functions. You are given a question and a set of "
    "possible functions. Based on the question, you will need to make one or more "
    "function/tool calls to achieve the purpose. Your answers must be accurate, should "
    "not contain extraneous information, and should be executable. If no function call "
    "can be made, respond with an empty list."
)


def _category_filename(category: str) -> str:
    return f"BFCL_v3_{category}.json"


def load_bfcl_category(cfg: Any) -> List[Dict[str, Any]]:
    """加载某一 BFCL 类别的全部样本（jsonl 行列表）。

    配置项：category / bfcl_local_root（可选本地目录）/ hf_path（默认官方 Hub 仓库）。
    """
    category = str(getattr(cfg, "category", "simple"))
    if category not in BFCL_ALL_CATEGORIES:
        raise ValueError(
            f"未知 BFCL 类别: {category!r}，可选: {sorted(BFCL_ALL_CATEGORIES)}"
        )
    fname = _category_filename(category)
    local_root = getattr(cfg, "bfcl_local_root", None)

    if local_root:
        fpath = os.path.join(os.path.expanduser(str(local_root)), fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"bfcl_local_root 下未找到 {fname}: {fpath}"
                f"（可用 scripts/download_bfcl_dataset.py 预先下载）"
            )
        rows: List[Dict[str, Any]] = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    hf_path = str(getattr(cfg, "hf_path", None) or BFCL_HF_PATH)
    from datasets import Dataset, load_dataset  # 延迟导入：离线/纯解析场景无需 datasets

    ds: Dataset = load_dataset(hf_path, data_files=fname, split="train")
    return [dict(zip(ds.column_names, vals)) for vals in zip(*ds)]


def _format_function_docs(functions: Any) -> str:
    """将 function 字段（list of {"name","description","parameters"}）格式化为文档。"""
    if isinstance(functions, str):
        return functions
    docs = []
    for fn in functions or []:
        docs.append(json.dumps(fn, ensure_ascii=False, indent=2))
    return "\n".join(docs)


def build_bfcl_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """构造 chat 消息：system（官方措辞）+ user（函数文档 + 问题）。"""
    func_doc = _format_function_docs(row.get("function"))
    question = str(row.get("question", "")).strip()
    user = (
        "Given the following functions, please respond to the user's question with "
        "function call(s) that best achieve the purpose.\n\n"
        f"Possible functions:\n{func_doc}\n\n"
        f"User question:\n{question}\n\n"
        "Respond with the function call(s) as a JSON list of objects "
        '[{"name": "<function_name>", "arguments": {...}}]. '
        "If no function call can be made, respond with an empty list []."
    )
    return [
        {"role": "system", "content": BFCL_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def build_bfcl_plain_prompt(row: Dict[str, Any]) -> str:
    """无 chat template 时的纯文本拼接版本。"""
    msgs = build_bfcl_messages(row)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
