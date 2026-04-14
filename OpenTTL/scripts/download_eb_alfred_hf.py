#!/usr/bin/env python3
"""从 Hugging Face 下载 EmbodiedBench/EB-ALFRED 数据。

支持：
  - ``HF_ENDPOINT=https://hf-mirror.com`` 镜像；
  - 全量模式：列举全部文件后 **分批 + 并发** 下载，**每文件重试 + 多轮补全**，避免一次 429 即整体失败；
  - 可选 ``hf_transfer`` 加速（``EB_USE_HF_TRANSFER=1``）；
  - ``single``：按 splits 只下一条任务（联调）。

环境变量（可选）：
  EB_BATCH_SIZE, EB_MAX_WORKERS, EB_MAX_ROUNDS, EB_PER_FILE_RETRIES,
  EB_INTER_BATCH_SLEEP, EB_USE_HF_TRANSFER, HF_HUB_ENABLE_HF_TRANSFER
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

REPO_ID = "EmbodiedBench/EB-ALFRED"


def _hub_endpoint() -> str | None:
    v = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    return v or None


def _patch_hf_pagination_rewrite_to_endpoint() -> None:
    """分页 ``Link: next`` 指向 huggingface.co 时需改写到 HF_ENDPOINT。"""
    mirror = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    if not mirror:
        return
    from huggingface_hub.utils import _pagination as pag
    from huggingface_hub.utils._http import fix_hf_endpoint_in_url

    if getattr(pag, "_openttl_rewrite_pagination", False):
        return

    def _paginate(path: str, params: dict, headers: dict):
        path = fix_hf_endpoint_in_url(path, mirror)
        session = pag.get_session()
        r = session.get(path, params=params, headers=headers)
        pag.hf_raise_for_status(r)
        yield from r.json()
        next_page = pag._get_next_page(r)
        while next_page is not None:
            next_page = fix_hf_endpoint_in_url(next_page, mirror)
            pag.logger.debug("Pagination next: %s", next_page)
            r = pag.http_backoff("GET", next_page, headers=headers)
            pag.hf_raise_for_status(r)
            yield from r.json()
            next_page = pag._get_next_page(r)

    pag.paginate = _paginate  # type: ignore[assignment]
    import huggingface_hub.utils as hub_utils

    hub_utils.paginate = _paginate  # type: ignore[assignment]
    import huggingface_hub.hf_api as hf_api_mod

    hf_api_mod.paginate = _paginate  # type: ignore[assignment]
    setattr(pag, "_openttl_rewrite_pagination", True)


def _repo_files_under_prefix(api, prefix: str) -> list[str]:
    from huggingface_hub.hf_api import RepoFile

    out: list[str] = []
    for item in api.list_repo_tree(
        REPO_ID,
        path_in_repo=prefix,
        recursive=True,
        repo_type="dataset",
    ):
        if isinstance(item, RepoFile):
            out.append(item.path)
    return sorted(out)


def _local_file_path(dest: Path, rel: str) -> Path:
    return dest / rel


def _file_looks_complete(path: Path, min_bytes: int = 64) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _download_one_file(
    rel: str,
    dest: Path,
    endpoint: str | None,
    max_attempts: int,
    base_sleep: float,
) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    import httpx

    target = _local_file_path(dest, rel)
    if _file_looks_complete(target):
        return

    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=rel,
                repo_type="dataset",
                local_dir=str(dest),
                endpoint=endpoint,
            )
            if _file_looks_complete(target):
                return
            last = RuntimeError(f"下载后仍不完整: {target}")
        except (HfHubHTTPError, httpx.TransportError, OSError) as e:
            last = e
            sleep_s = min(base_sleep * (2 ** min(attempt, 10)), 300.0)
            if isinstance(e, HfHubHTTPError) and getattr(e, "response", None) is not None:
                code = e.response.status_code
                if code == 429:
                    ra = e.response.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = max(sleep_s, float(ra))
                        except ValueError:
                            pass
                if code in (502, 503, 504):
                    sleep_s = max(sleep_s, base_sleep * (attempt + 1))
            time.sleep(sleep_s)
        except BaseException as e:
            last = e
            time.sleep(base_sleep * (attempt + 1))

    assert last is not None
    raise last


def _download_files(rel_paths: Iterable[str], dest: Path, endpoint: str | None) -> int:
    n = 0
    dest.mkdir(parents=True, exist_ok=True)
    for rel in rel_paths:
        _download_one_file(rel, dest, endpoint, max_attempts=8, base_sleep=1.5)
        n += 1
    return n


def _download_full_resumable(
    dest: Path,
    endpoint: str | None,
    api,
    *,
    batch_size: int,
    max_workers: int,
    max_rounds: int,
    per_file_retries: int,
    inter_batch_sleep: float,
    base_sleep: float,
) -> None:
    from huggingface_hub.hf_api import RepoFile

    dest.mkdir(parents=True, exist_ok=True)
    print("列举远端文件…", file=sys.stderr)
    all_rels: list[str] = []
    for item in api.list_repo_tree(REPO_ID, recursive=True, repo_type="dataset"):
        if isinstance(item, RepoFile):
            all_rels.append(item.path)

    def still_missing() -> list[str]:
        return [r for r in all_rels if not _file_looks_complete(_local_file_path(dest, r))]

    missing = still_missing()
    print(f"共 {len(all_rels)} 个文件，待下载 {len(missing)}", file=sys.stderr)

    for rnd in range(1, max_rounds + 1):
        missing = still_missing()
        if not missing:
            print(f"第 {rnd} 轮检查：已全部就绪", file=sys.stderr)
            return
        print(f"===== 第 {rnd}/{max_rounds} 轮，待完成 {len(missing)} 个文件 =====", file=sys.stderr)

        for bi in range(0, len(missing), batch_size):
            batch = missing[bi : bi + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        _download_one_file,
                        rel,
                        dest,
                        endpoint,
                        per_file_retries,
                        base_sleep,
                    ): rel
                    for rel in batch
                }
                for fut in as_completed(futures):
                    rel = futures[fut]
                    try:
                        fut.result()
                    except BaseException as e:
                        print(f"[失败] {rel}: {e!r}", file=sys.stderr)
            if inter_batch_sleep > 0:
                time.sleep(inter_batch_sleep)

    missing = still_missing()
    if missing:
        raise RuntimeError(f"经过 {max_rounds} 轮仍有 {len(missing)} 个文件未就绪（示例: {missing[:3]}）")


def _default_embodiedbench_src() -> Path:
    root = Path(__file__).resolve().parents[1]
    return (root / "third_party" / "EmbodiedBench").resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download EB-ALFRED (EmbodiedBench) from Hugging Face Hub.")
    p.add_argument(
        "--dest",
        default=os.environ.get("EB_DEST", str(Path(os.environ.get("TMPDIR", "/tmp")) / "eb_alfred_json_2.1.0")),
        help="Local directory (becomes json_2.1.0 root: <dest>/<task>/... ).",
    )
    p.add_argument(
        "--embodiedbench-src",
        default=os.environ.get("EMBODIEDBENCH_SRC", str(_default_embodiedbench_src())),
        help="EmbodiedBench checkout (for splits.json in single mode).",
    )
    p.add_argument(
        "--mode",
        choices=("full", "single"),
        default=os.environ.get("EB_DOWNLOAD_MODE", "full"),
        help="full: entire dataset (batched+retry); single: one task from splits.json.",
    )
    p.add_argument(
        "--eval-set",
        default=os.environ.get("EB_EVAL_SET", "base"),
        help="Split name in splits.json (single mode).",
    )
    p.add_argument(
        "--task-index",
        type=int,
        default=int(os.environ.get("EB_TASK_INDEX", "0")),
        help="Index into splits[eval_set] (single mode).",
    )
    p.add_argument(
        "--task-path",
        default=os.environ.get("EB_TASK_PATH", "").strip(),
        help="If set, override splits and download this repo-relative task folder (e.g. .../trial_T...).",
    )
    p.add_argument(
        "--symlink",
        action="store_true",
        default=os.environ.get("EB_SYMLINK", "").strip() in ("1", "true", "yes"),
        help="After download, ln -sfn DEST -> <embodiedbench>/embodiedbench/envs/eb_alfred/data/json_2.1.0",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EB_BATCH_SIZE", "32")),
        help="每批提交的文件数。",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("EB_MAX_WORKERS", "4")),
        help="每批内并发线程数；镜像站可配合 EB_INTER_BATCH_SLEEP 防 429。",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=int(os.environ.get("EB_MAX_ROUNDS", "50")),
        help="多轮扫尾：每轮重新检查缺失文件并再下，直到全部完成或达到上限。",
    )
    p.add_argument(
        "--per-file-retries",
        type=int,
        default=int(os.environ.get("EB_PER_FILE_RETRIES", "16")),
        help="单个文件失败时的最大重试次数（含指数退避）。",
    )
    p.add_argument(
        "--inter-batch-sleep",
        type=float,
        default=float(os.environ.get("EB_INTER_BATCH_SLEEP", "0.75")),
        help="批次之间的休眠秒数，缓解镜像限流。",
    )
    p.add_argument(
        "--retry-base-sleep",
        type=float,
        default=float(os.environ.get("EB_RETRY_BASE_SLEEP", "1.5")),
        help="单文件重试的基础休眠（秒），实际为指数退避。",
    )
    p.add_argument(
        "--use-hf-transfer",
        action="store_true",
        default=os.environ.get("EB_USE_HF_TRANSFER", "").strip().lower() in ("1", "true", "yes"),
        help="启用 hf_transfer（需 pip install hf_transfer）；部分网络/镜像下可显著加速。",
    )
    args = p.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    eb_src = Path(args.embodiedbench_src).expanduser().resolve()

    if args.use_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            print("警告: 已请求 hf_transfer 但未安装，执行: pip install hf_transfer", file=sys.stderr)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    print(f"HF_ENDPOINT={hf_endpoint or '(default hub)'}", file=sys.stderr)
    print(f"mode={args.mode} dest={dest}", file=sys.stderr)
    print(
        f"batch_size={args.batch_size} max_workers={args.max_workers} "
        f"max_rounds={args.max_rounds} per_file_retries={args.per_file_retries} "
        f"inter_batch_sleep={args.inter_batch_sleep} hf_transfer={args.use_hf_transfer}",
        file=sys.stderr,
    )

    _patch_hf_pagination_rewrite_to_endpoint()

    from huggingface_hub import HfApi

    endpoint = _hub_endpoint()
    api = HfApi(endpoint=endpoint)

    if args.mode == "full":
        _download_full_resumable(
            dest,
            endpoint,
            api,
            batch_size=max(1, args.batch_size),
            max_workers=max(1, args.max_workers),
            max_rounds=max(1, args.max_rounds),
            per_file_retries=max(1, args.per_file_retries),
            inter_batch_sleep=max(0.0, args.inter_batch_sleep),
            base_sleep=max(0.1, args.retry_base_sleep),
        )
        print(f"Downloaded full dataset to {dest}")
    else:
        if args.task_path:
            task_path = args.task_path.strip().strip("/")
        else:
            splits_path = eb_src / "embodiedbench" / "envs" / "eb_alfred" / "data" / "splits" / "splits.json"
            if not splits_path.is_file():
                print(f"错误: 找不到 {splits_path}，请设置 --embodiedbench-src 或先克隆 EmbodiedBench。", file=sys.stderr)
                return 1
            data = json.loads(splits_path.read_text(encoding="utf-8"))
            tasks = data.get(args.eval_set)
            if not isinstance(tasks, list) or not tasks:
                print(f"错误: splits 中无列表 eval_set={args.eval_set!r}", file=sys.stderr)
                return 1
            if args.task_index < 0 or args.task_index >= len(tasks):
                print(f"错误: task_index 越界: {args.task_index} (共 {len(tasks)} 条)", file=sys.stderr)
                return 1
            task_path = str(tasks[args.task_index]["task"]).strip().strip("/")

        files = _repo_files_under_prefix(api, task_path)
        files = [f for f in files if not f.endswith("/")]
        if not files:
            print(f"错误: Hub 上未找到任务路径前缀 {task_path!r}", file=sys.stderr)
            return 1
        n = _download_files(files, dest, endpoint)
        print(f"Downloaded {n} files for task {task_path!r} -> {dest}")

    if args.symlink:
        link = eb_src / "embodiedbench" / "envs" / "eb_alfred" / "data" / "json_2.1.0"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() and not link.is_symlink():
            if link.is_dir():
                print(f"错误: 已存在真实目录（非符号链接），请手动处理: {link}", file=sys.stderr)
                return 1
            print(f"错误: 路径已被文件占用: {link}", file=sys.stderr)
            return 1
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(dest, target_is_directory=True)
        print(f"Symlinked {link} -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
