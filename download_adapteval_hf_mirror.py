#!/usr/bin/env python3
"""
从 hf-mirror 下载 Jinwu01/AdaptEval:
https://hf-mirror.com/datasets/Jinwu01/AdaptEval

用法:
  pip install huggingface_hub
  python download_adapteval_hf_mirror.py
  python download_adapteval_hf_mirror.py --local-dir /path/to/save
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ID = "Jinwu01/AdaptEval"
_HF_MIRROR = "https://hf-mirror.com"
_DEFAULT_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "AdaptEval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Jinwu01/AdaptEval from hf-mirror.")
    p.add_argument(
        "--local-dir",
        default=_DEFAULT_LOCAL_DIR,
        help=f"保存目录（默认: {_DEFAULT_LOCAL_DIR}）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少 huggingface_hub，请先安装: pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Source: {_HF_MIRROR}/datasets/{_REPO_ID}")
    print(f"local_dir={args.local_dir}")

    path = snapshot_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        local_dir=args.local_dir,
    )
    print(f"Done. Path: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
