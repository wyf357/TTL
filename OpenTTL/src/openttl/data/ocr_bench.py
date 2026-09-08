"""OCRBench / OmniDocBench 数据加载与 prompt 构建。

判分口径与 lmms-eval 完全一致：本模块只做数据/prompt，判分函数在评测脚本里
直接 import ``lmms_eval.tasks.{ocrbench,omnidocbench}.utils``。

- OCRBench: ``echo840/OCRBench`` test（1000 题；question + image，短答案）
- OmniDocBench: ``ouyanglinke/OmniDocBench_tsv`` train（981 页；image 为 base64
  字符串，answer 为标注 JSON；prompt 固定为 doc→markdown 指令）
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from PIL import Image

# 与 lmms_eval.tasks.omnidocbench.utils._DOC_TO_MARKDOWN_PROMPT 保持一致
OMNIDOC_PROMPT = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Output ONLY the converted markdown — no explanations, no thinking process, no commentary. "
    "Rules: "
    "1. Recognize all text accurately and convert to Markdown. "
    "2. Convert mathematical formulas to LaTeX (inline: $...$, display: $$...$$). "
    "3. Convert tables to HTML format. "
    "4. Ignore figures and images. "
    "5. Maintain the original document structure and reading order."
)


def decode_omni_image(image_obj: Any) -> Image.Image:
    """OmniDocBench_tsv 的 image 列是 base64 字符串；也兼容 PIL/bytes。"""
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    if isinstance(image_obj, str):
        return Image.open(io.BytesIO(base64.b64decode(image_obj))).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image_obj)}")


def load_ocrbench(split: str = "test") -> Any:
    from datasets import load_dataset

    return load_dataset("echo840/OCRBench", split=split)


def load_omnidocbench(split: str = "train") -> Any:
    from datasets import load_dataset

    return load_dataset("ouyanglinke/OmniDocBench_tsv", split=split)


def ocrbench_prompt(row: Dict[str, Any]) -> str:
    """与 lmms-eval 默认（pre/post_prompt 均为空）一致。"""
    return str(row["question"]).strip()


def omnidocbench_prompt(row: Dict[str, Any]) -> str:  # noqa: ARG001
    """lmms-eval 的 doc_to_text 返回固定指令，与页面无关。"""
    return OMNIDOC_PROMPT
