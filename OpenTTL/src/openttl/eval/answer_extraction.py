"""多选题答案抽取（A/B/C/D），兼容 Qwen3.5 thinking 模式。

逻辑与 evaluations/run_erqa.py 中的实现保持一致，供多个多模态
选择题评测（ERQA、MMStar 等）共用。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Qwen3.5 thinking-mode tags (constructed via unicode escapes to avoid
# XML-parsing issues in tool calls)
_THINK_CLOSE = "</think>"


def strip_thinking_block(text: str) -> str:
    """Remove Qwen3.5 thinking block from response text.

    Qwen3.5 generates a thinking block before the actual answer.
    We detect the closing tag and return only the text that follows it.
    If no thinking block is found, the original text is returned unchanged.
    """
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE)[-1].strip()
    return text


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter (A, B, C, or D) from model response.

    Handles Qwen3.5 thinking mode by first stripping the thinking block,
    then searching for the answer letter in the remaining text.
    """
    # First strip any thinking block
    answer_part = strip_thinking_block(response)

    # Prefer the last "Answer: X" in the visible part: few-shot prompts may list several
    # "Answer: …" lines; the instruction requires the final line to be the true choice.
    _ans_marks = re.findall(r"[Aa]nswer\s*[:\uff1a]\s*([A-D])", answer_part)
    if _ans_marks:
        return _ans_marks[-1].upper()

    # Try structured patterns first on the answer part
    answer_patterns = [
        r'[Tt]he' + r'\s+answer' + r'\s+is' + r'\s+([A-D])',
        r'[Oo]ption' + r'\s+([A-D])' + r'\b',
        r'[Cc]hoice' + r'\s+([A-D])' + r'\b',
        r'^' + r'\s*([A-D])' + r'\s*$',
        r'^' + r'\s*([A-D])' + r'[\.\uff0c,\s]',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, answer_part, re.MULTILINE)
        if match:
            return match.group(1).upper()

    # If no pattern matches in answer part, try the first 100 chars
    first_part = answer_part[:100]
    for letter in ['A', 'B', 'C', 'D']:
        if letter in first_part:
            return letter

    # Any "Answer: X" in the full raw response (thinking + visible); use last match.
    _ans_all = re.findall(r"[Aa]nswer\s*[:\uff1a]\s*([A-D])", response)
    if _ans_all:
        return _ans_all[-1].upper()

    # Last resort: scan the ENTIRE response (including thinking)
    # Add patterns that commonly appear inside the thinking block itself,
    # where the model states its conclusion before formatting the answer.
    thinking_patterns = answer_patterns + [
        r'[Tt]herefore[\s,]+(?:the answer is\s+)?([A-D])',
        r'[Ss]o(?:\s+the answer is)?\s+([A-D])',
        r'[Aa]ccordingly[\s,]+(?:the answer is\s+)?([A-D])',
        r'[Tt]hus[\s,]+(?:the answer is\s+)?([A-D])',
        r'[Ii] conclude(?:\s+that)?(?:\s+the answer is)?\s+([A-D])',
    ]
    for pattern in thinking_patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return match.group(1).upper()

    return None


def slugify_model_for_csv(name: str) -> str:
    """Normalize Hydra model group name or checkpoint dirname for CSV basename.

    Examples: ``qwen35_2b`` -> ``qwen35_2B``; ``Qwen3.5-2B`` path segment -> ``qwen3_5_2B``.
    """
    s = name.strip()
    if "/" in s or "\\" in s:
        s = Path(s).name
    s = s.lower().replace("-", "_")
    m = re.match(r"^(.+_)(\d+)(b)$", s)
    if m:
        return m.group(1) + m.group(2) + "B"
    return s
