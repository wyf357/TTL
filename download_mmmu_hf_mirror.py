#!/usr/bin/env python3
"""
从 Hugging Face 镜像（hf-mirror）下载 MMMU 数据集仓库到本地目录。

原站: https://huggingface.co/datasets/MMMU/MMMU

镜像通过环境变量 HF_ENDPOINT 指向 https://hf-mirror.com（与 huggingface_hub 官方约定一致）。

用法:
  pip install huggingface_hub
  python download_mmmu_hf_mirror.py
  （在 AutoDL 上默认下载到 /root/autodl-tmp/MMMU/；本地无数据盘时为当前目录下 MMMU/）

  # 与 lmms-eval 的 mmmu_val 任务对齐（推荐用于标准 MMMU 评测）:
  python download_mmmu_hf_mirror.py --preset lmms-eval
  # 等价于 --repo-id lmms-lab/MMMU --local-dir .../lmms-lab-MMMU/

环境变量（可选）:
  HF_ENDPOINT               默认本脚本会设为 https://hf-mirror.com；若你已在外部设置则尊重已有值。
  HF_TOKEN                  仅私有或需鉴权时使用；MMMU/MMMU 为公开数据集，一般不必设置。
  HF_HUB_DOWNLOAD_TIMEOUT   单次 HTTP 下载超时（秒）；脚本默认设为 600，镜像慢时可再加大。

中断后可直接重新运行同一命令；huggingface_hub 会对已存在/未下完的文件尽量断点续传。

镜像在高并发时容易出现 TLS 握手超时（ConnectTimeout），默认已降低并行线程数并带自动重试。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ID = "MMMU/MMMU"
# lmms-eval 任务 mmmu_val / mmmu_test 使用 dataset_path: lmms-lab/MMMU（与上者可能不同步，评测请优先下此仓库）
_LMMS_EVAL_MMU_REPO_ID = "lmms-lab/MMMU"
_DEFAULT_MIRROR = "https://hf-mirror.com"
# AutoDL 数据盘通常为 /root/autodl-tmp（连字符）；少数环境为 /root/autodl/tmp
_CANDIDATE_BASES = ("/root/autodl-tmp", "/root/autodl/tmp")
# 在数据盘下单独建目录存放本数据集
_DATASET_DIRNAME = "MMMU"


def _configure_hub_for_mirror() -> None:
    """镜像不稳定时：拉长超时、避免过短默认导致握手/读超时。"""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def _is_transient_download_error(err: BaseException) -> bool:
    """网络抖动、镜像限流、TLS 握手超时等可重试；其它错误交给调用方。"""
    if isinstance(err, (BrokenPipeError, ConnectionResetError, TimeoutError, ConnectionError)):
        return True
    mod = getattr(type(err), "__module__", "") or ""
    if mod.startswith(("httpx", "httpcore")):
        return True
    name = type(err).__name__
    return any(
        x in name
        for x in (
            "Timeout",
            "Connect",
            "Connection",
            "Protocol",
            "Reset",
            "Temporary",
        )
    )


def _default_local_dir() -> str:
    for base in _CANDIDATE_BASES:
        if os.path.isdir(base):
            return os.path.join(base, _DATASET_DIRNAME)
    return os.path.join(os.getcwd(), _DATASET_DIRNAME)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Download {_REPO_ID} via HF mirror (hf-mirror.com)."
    )
    p.add_argument(
        "--preset",
        choices=("default", "lmms-eval"),
        default="default",
        help=(
            "lmms-eval: 下载 lmms-eval 任务使用的 lmms-lab/MMMU，"
            "并默认保存到数据盘下 lmms-lab-MMMU/（与 MMMU/MMMU 目录区分）"
        ),
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face 数据集 repo id（不设则由 --preset 决定）",
    )
    p.add_argument(
        "--local-dir",
        default=None,
        help=(
            "保存目录（默认: 若存在数据盘则在盘内子目录 "
            f"{_DATASET_DIRNAME}/ ，例如 /root/autodl-tmp/{_DATASET_DIRNAME}/；"
            "否则为当前目录下同名子目录；lmms-eval preset 时为 .../lmms-lab-MMMU/）"
        ),
    )
    p.add_argument(
        "--endpoint",
        default=_DEFAULT_MIRROR,
        help="HF 镜像根 URL（默认 https://hf-mirror.com）",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="并行下载线程数（镜像建议 1～2，过高易触发 ConnectTimeout）",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=12,
        help="整次 snapshot_download 失败后的最大重试次数（每次会从头调度未完成的文件）",
    )
    p.add_argument(
        "--retry-wait",
        type=int,
        default=25,
        help="两次重试之间的基础等待秒数（实际等待会随重试次数略增）",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="可选：commit / 分支 / 标签",
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    preset = args.preset
    repo_id = args.repo_id
    if repo_id is None:
        repo_id = _LMMS_EVAL_MMU_REPO_ID if preset == "lmms-eval" else _REPO_ID
    local_dir = args.local_dir
    if local_dir is None:
        if preset == "lmms-eval":
            for base in _CANDIDATE_BASES:
                if os.path.isdir(base):
                    local_dir = os.path.join(base, "lmms-lab-MMMU")
                    break
            if local_dir is None:
                local_dir = os.path.join(os.getcwd(), "lmms-lab-MMMU")
        else:
            local_dir = _default_local_dir()
    return repo_id, local_dir


def main() -> int:
    args = parse_args()
    args.repo_id, args.local_dir = _resolve_paths(args)

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
    _configure_hub_for_mirror()

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
    print(f"preset={args.preset}")
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

    retries = max(1, args.retries)
    for attempt in range(1, retries + 1):
        try:
            path = snapshot_download(**kwargs)
            print(f"Done. Path: {path}")
            return 0
        except BaseException as e:
            transient = _is_transient_download_error(e)
            if not transient or attempt >= retries:
                raise
            wait = min(180, args.retry_wait + 5 * (attempt - 1))
            print(
                f"[{attempt}/{retries}] 下载中断: {type(e).__name__}: {e}\n"
                f"    {wait}s 后自动重试（已下载内容会尽量续传）…",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError("unreachable: snapshot_download should return or raise")


if __name__ == "__main__":
    raise SystemExit(main())
