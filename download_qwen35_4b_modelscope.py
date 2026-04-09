#!/usr/bin/env python3
"""
从魔搭 ModelScope 拉取 Qwen/Qwen3.5-4B（与网页一致）:
https://www.modelscope.ai/models/Qwen/Qwen3.5-4B

用法:
  pip install modelscope
  python download_qwen35_4b_modelscope.py
  python download_qwen35_4b_modelscope.py --local-dir /root/autodl-tmp/其它目录

环境变量（可选）:
  MODELSCOPE_API_TOKEN  私有或需鉴权仓库时使用；本模型为公开，一般不必设置。
"""

from __future__ import annotations

import argparse
import os
import sys

# AutoDL 数据盘通常为 /root/autodl-tmp
_AUTODL_TMP = "/root/autodl-tmp"
_DEFAULT_LOCAL_DIR = os.path.join(_AUTODL_TMP, "Qwen3.5-4B")
_MODELSCOPE_PAGE = "https://www.modelscope.ai/models/Qwen/Qwen3.5-4B"


def _default_local_dir() -> str | None:
    if os.path.isdir(_AUTODL_TMP):
        return _DEFAULT_LOCAL_DIR
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Qwen/Qwen3.5-4B from ModelScope (魔搭)."
    )
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-4B",
        help="ModelScope 模型 ID（默认与网页一致: Qwen/Qwen3.5-4B）",
    )
    p.add_argument(
        "--local-dir",
        default=_default_local_dir(),
        help=(
            "保存目录；在检测到 AutoDL 数据盘时为 "
            f"{_DEFAULT_LOCAL_DIR}，否则使用 ModelScope 默认缓存目录"
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行下载线程数",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="可选：分支或标签名（默认使用模型默认版本）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            print(
                "缺少 modelscope，请先安装: pip install modelscope",
                file=sys.stderr,
            )
            return 1

    print(f"Source: ModelScope {_MODELSCOPE_PAGE}")
    print(f"model_id={args.model_id}")

    kwargs = {
        "model_id": args.model_id,
        "max_workers": args.max_workers,
    }
    if args.local_dir:
        kwargs["local_dir"] = args.local_dir
    if args.revision:
        kwargs["revision"] = args.revision

    path = snapshot_download(**kwargs)
    print(f"Done. Path: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
