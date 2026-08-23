"""下载 BFCL v3 数据集 json 文件到本地目录（离线环境评测用）。

用法：
  python scripts/download_bfcl_dataset.py --out /root/autodl-tmp/bfcl
  python scripts/download_bfcl_dataset.py --out /root/autodl-tmp/bfcl --categories simple multiple
国内无外网时可先 export HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> None:
    from huggingface_hub import hf_hub_download

    from openttl.data.bfcl import BFCL_ALL_CATEGORIES, BFCL_HF_PATH

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="本地输出目录")
    ap.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help=f"要下载的类别（默认全部）；可选: {sorted(BFCL_ALL_CATEGORIES)}",
    )
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    cats = args.categories or list(BFCL_ALL_CATEGORIES)

    def _link(src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"[skip] {dest} 已存在")
            return
        try:
            os.symlink(src, dest)
        except OSError:
            import shutil

            shutil.copy(src, dest)
        print(f"[ok] {dest.name} -> {dest}")

    for cat in cats:
        fname = f"BFCL_v3_{cat}.json"
        path = hf_hub_download(repo_id=BFCL_HF_PATH, filename=fname, repo_type="dataset")
        _link(path, out / fname)
        gt_rel = f"possible_answer/BFCL_v3_{cat}.json"
        try:
            gt_path = hf_hub_download(
                repo_id=BFCL_HF_PATH, filename=gt_rel, repo_type="dataset"
            )
            _link(gt_path, out / gt_rel)
        except Exception as e:
            print(f"[warn] 无 possible_answer（{cat}）: {e}")


if __name__ == "__main__":
    main()
