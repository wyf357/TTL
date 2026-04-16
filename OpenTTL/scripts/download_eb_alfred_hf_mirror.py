#!/usr/bin/env python3
"""从 Hugging Face 国内镜像 (hf-mirror) 下载 EmbodiedBench/EB-ALFRED 数据集到项目内路径。

镜像站: https://hf-mirror.com/datasets/EmbodiedBench/EB-ALFRED/tree/main

默认将数据写入::
    <OpenTTL>/third_party/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0

下载策略（由 ``download_eb_alfred_hf.py`` 实现）：
  - 分批并发、多轮扫尾、单文件指数退避重试，适合大仓库与不稳定网络；
  - 默认 ``HF_ENDPOINT=https://hf-mirror.com``，并关闭 ``hf_transfer``（与部分镜像不兼容，可用环境变量显式打开）。

用法::

    pip install huggingface_hub httpx
    python scripts/download_eb_alfred_hf_mirror.py

    # 与原版脚本相同的参数均支持，例如仅下一条任务::
    EB_DOWNLOAD_MODE=single EB_TASK_INDEX=0 python scripts/download_eb_alfred_hf_mirror.py

    # 自定义项目根或目标目录::
    python scripts/download_eb_alfred_hf_mirror.py --dest /data/eb/json_2.1.0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# OpenTTL 仓库根（scripts/ 的上一级）
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DEST = _REPO_ROOT / "third_party/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0"
_DEFAULT_EB_SRC = _REPO_ROOT / "third_party/EmbodiedBench"


def _mirror_env_defaults() -> None:
    """镜像端点；避免 hf_transfer 在镜像上出错。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 未显式设置时关闭 hf_transfer（与 hf-mirror 兼容性更好）
    if not (os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or "").strip():
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def _argv_has_option(argv: list[str], name: str) -> bool:
    """检查 ``--name`` 或 ``--name=`` 是否已出现。"""
    prefix = f"{name}="
    for a in argv:
        if a == name or a.startswith(prefix):
            return True
    return False


def _inject_project_defaults(argv: list[str]) -> list[str]:
    """未指定 --dest / --embodiedbench-src 时注入项目内默认路径。"""
    out = list(argv)
    if not _argv_has_option(out, "--dest"):
        out = ["--dest", str(_DEFAULT_DEST), *out]
    if not _argv_has_option(out, "--embodiedbench-src"):
        out = ["--embodiedbench-src", str(_DEFAULT_EB_SRC), *out]
    return out


def main() -> int:
    _mirror_env_defaults()
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from download_eb_alfred_hf import main as hf_main

    argv = _inject_project_defaults(sys.argv[1:])
    print(
        f"[eb-alfred-mirror] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')} "
        f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '')}",
        file=sys.stderr,
    )
    print(f"[eb-alfred-mirror] 默认数据根（可被 --dest 覆盖）: {_DEFAULT_DEST}", file=sys.stderr)
    return hf_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
