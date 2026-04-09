#!/usr/bin/env python3
"""
从 Hugging Face 镜像（hf-mirror）下载 MMLU 数据集仓库到本地目录。

原站: https://huggingface.co/datasets/cais/mmlu

镜像通过环境变量 HF_ENDPOINT 指向 https://hf-mirror.com（与 huggingface_hub / datasets 官方约定一致）。

用法:
  pip install huggingface_hub
  python download_mmlu_hf_mirror.py
  python download_mmlu_hf_mirror.py --local-dir /root/autodl-tmp/mmlu

环境变量（可选）:
  HF_ENDPOINT   默认本脚本会设为 https://hf-mirror.com；若你已在外部设置则尊重已有值。
  HF_TOKEN      仅私有或需鉴权时使用；cais/mmlu 为公开数据集，一般不必设置。
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ID = "cais/mmlu"
_DEFAULT_MIRROR = "https://hf-mirror.com"
# AutoDL 数据盘通常为 /root/autodl-tmp（连字符）；少数环境为 /root/autodl/tmp
_CANDIDATE_BASES = ("/root/autodl-tmp", "/root/autodl/tmp")


def _default_local_dir() -> str:
    for base in _CANDIDATE_BASES:
        if os.path.isdir(base):
            return os.path.join(base, "mmlu")
    return os.path.join(os.getcwd(), "mmlu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Download {_REPO_ID} via HF mirror (hf-mirror.com)."
    )
    p.add_argument(
        "--repo-id",
        default=_REPO_ID,
        help="Hugging Face 数据集 repo id",
    )
    p.add_argument(
        "--local-dir",
        default=_default_local_dir(),
        help="保存目录（默认: 若存在 /root/autodl-tmp 或 /root/autodl/tmp 则在其下 mmlu/）",
    )
    p.add_argument(
        "--endpoint",
        default=_DEFAULT_MIRROR,
        help="HF 镜像根 URL（默认 https://hf-mirror.com）",
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
        help="可选：commit / 分支 / 标签",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "缺少 huggingface_hub，请先安装: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"repo_id={args.repo_id} (repo_type=dataset)")
    print(f"local_dir={args.local_dir}")

    kwargs = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": args.local_dir,
        "max_workers": args.max_workers,
    }
    if args.revision:
        kwargs["revision"] = args.revision

    path = snapshot_download(**kwargs)
    print(f"Done. Path: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
