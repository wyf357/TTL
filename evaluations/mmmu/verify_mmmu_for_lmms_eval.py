#!/usr/bin/env python3
"""
检查本机能否通过 Hugging Face datasets 读取 lmms-eval 使用的 MMMU 数据源（lmms-lab/MMMU）。

用法:
  pip install datasets
  export HF_HOME=/root/autodl-tmp/hf   # 可选，大磁盘
  python verify_mmmu_for_lmms_eval.py

离线（已缓存）:
  export HF_DATASETS_OFFLINE=1
  python verify_mmmu_for_lmms_eval.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "(未设置)")
    dsc = os.environ.get("HF_DATASETS_CACHE", "(未设置，将随 HF_HOME 默认)")
    print(f"HF_HOME={hf_home}")
    print(f"HF_DATASETS_CACHE={dsc}")
    print(f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', '0')}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: pip install datasets", file=sys.stderr)
        return 1

    # MMMU 按学科分 config；任取一个验证可读
    config = "Accounting"
    try:
        ds = load_dataset(
            "lmms-lab/MMMU",
            config,
            split="validation",
            trust_remote_code=True,
        )
        n = len(ds)
        row = ds[0]
        keys = list(row.keys())[:12]
        print(f"OK: config={config!r} split=validation len={n} sample_keys={keys}")
        return 0
    except Exception as e:
        print(
            "读取失败。若尚未缓存，请先联网运行一次本脚本，或执行:\n"
            "  python ../../download_mmmu_hf_mirror.py --preset lmms-eval\n"
            "并将 HF_HOME 指向含 datasets 缓存的目录（datasets 仍会按 repo 名建缓存，"
            "snapshot 目录与 load_dataset 缓存不同，评测前建议至少成功 load 一次）。\n"
            f"错误: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
