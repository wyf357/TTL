"""MMLU 无标签适应：仅题干 + 四选项，不包含 answer。"""

from __future__ import annotations

import glob
import os
from typing import Any, List, Sequence, Union

from datasets import Dataset, DatasetDict, load_dataset


def _load_local_mmlu_table(files: Union[str, List[str]]) -> Dataset:
    """从本地 parquet 或 json/jsonl 加载（如 hf-mirror 快照落盘）。"""
    path = files[0] if isinstance(files, list) else files
    path_s = str(path).lower()
    if path_s.endswith(".jsonl") or path_s.endswith(".json"):
        raw = load_dataset("json", data_files=files)
    else:
        raw = load_dataset("parquet", data_files=files)
    if isinstance(raw, DatasetDict):
        return raw["train"] if "train" in raw else raw[next(iter(raw.keys()))]
    return raw


def _discover_local_parquets(local_root: str, hf_name: str, split: str) -> List[str]:
    """在 snapshot 目录下查找某学科、某 split 的 parquet（不访问 Hub）。"""
    root = os.path.expanduser(str(local_root))
    name = str(hf_name)
    split_l = str(split).lower()
    subject_dir = os.path.join(root, name)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(
            f"mmlu_local_root 下未找到学科目录: {subject_dir} "
            f"（请确认 snapshot 路径与 data.hf_name 一致）"
        )
    all_pq = sorted(
        glob.glob(os.path.join(subject_dir, "**", "*.parquet"), recursive=True)
    )
    if not all_pq:
        raise FileNotFoundError(f"目录中无 parquet 文件: {subject_dir}")

    def _basename(p: str) -> str:
        return os.path.basename(p).lower()

    tagged = [
        p
        for p in all_pq
        if _basename(p).startswith(split_l + "-")
        or f"/{split_l}/" in p.replace("\\", "/").lower()
    ]
    if not tagged:
        raise FileNotFoundError(
            f"在 {subject_dir} 下未找到 split={split!r} 的 parquet（例如 test-*.parquet）。"
            f" 已有文件示例: {all_pq[:3]!r}"
        )
    return tagged


def load_mmlu_source_dataset(data_cfg: Any) -> Dataset:
    """
    加载原始 MMLU 表（含 question / choices / answer），供适应或评测共用。

    优先级：mmlu_data_files > mmlu_local_root > Hub（需网络）。
    """
    path = getattr(data_cfg, "hf_path", None) or "cais/mmlu"
    name = getattr(data_cfg, "hf_name", None)
    split = str(getattr(data_cfg, "split", "test"))
    files = getattr(data_cfg, "mmlu_data_files", None)
    local_root = getattr(data_cfg, "mmlu_local_root", None)

    if files is not None:
        ds = _load_local_mmlu_table(files)
    elif local_root is not None:
        if not name:
            raise ValueError("使用 mmlu_local_root 时必须设置 data.hf_name（学科目录名）")
        pq = _discover_local_parquets(str(local_root), str(name), split)
        ds = _load_local_mmlu_table(pq)
    else:
        if not name:
            raise ValueError("从 Hub 加载 MMLU 需要 data.hf_name（例如 abstract_algebra）")
        ds = load_dataset(path, str(name), split=split)

    need = {"question", "choices"}
    missing = need - set(ds.column_names)
    if missing:
        raise ValueError(f"MMLU 缺少列 {missing}，当前为 {ds.column_names}")
    return ds


_LETTERS = ("A", "B", "C", "D")


def format_mmlu_prompt_no_answer(question: str, choices: Sequence[str]) -> str:
    q = str(question).strip()
    ch = [str(c).strip() for c in choices]
    if len(ch) != 4:
        raise ValueError(f"MMLU expects 4 choices, got {len(ch)}")
    lines = [f"Question: {q}"]
    for i, c in enumerate(ch):
        lines.append(f"{_LETTERS[i]}. {c}")
    return "\n".join(lines)


def _rows_to_text(batch: dict) -> dict:
    out: List[str] = []
    n = len(batch["question"])
    for i in range(n):
        out.append(
            format_mmlu_prompt_no_answer(batch["question"][i], batch["choices"][i])
        )
    return {"text": out}


def load_mmlu_unlabeled_raw(data_cfg: Any) -> Dataset:
    """
    加载 MMLU，映射为单列 text；丢弃 question/choices/answer 等原始列。

    配置项：
      hf_path / hf_name / split: Hub 加载时使用（无本地根目录时）
      mmlu_local_root: 本机 snapshot 根目录（与 hf_name 组成 …/hf_name/…/*.parquet），不访问 Hub
      mmlu_data_files: 显式 parquet / jsonl 路径或列表
      max_samples: 可选截断
    """
    ds = load_mmlu_source_dataset(data_cfg)

    mapped = ds.map(
        _rows_to_text,
        batched=True,
        remove_columns=[c for c in ds.column_names],
    )
    max_s = getattr(data_cfg, "max_samples", None)
    if max_s is not None:
        mapped = mapped.select(range(min(int(max_s), len(mapped))))
    return mapped
