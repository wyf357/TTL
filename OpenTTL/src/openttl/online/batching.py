from __future__ import annotations

from typing import Any, List, Optional

import torch

from openttl.adapters.base import ModelAdapter


def strategy_to_label_mode(strategy_name: str, *, prompt_only_tta: bool = False) -> str:
    """ERQA TTA: TENT/EATA/COME 用 ``none``; TLM 用整段 prompt 的 CLM（无标签、无 response 拼接）.

    ``prompt_only_tta=True`` (e.g. MMLU 非 TENT) → ``clm_full``; 否则 ``clm_response``（MMLU 等）。
    TLM 始终 ``clm_full``，见 :func:`strategy_suppresses_response`。
    """
    s = str(strategy_name or "tent").lower()
    if s in ("tent", "eata", "come"):
        return "none"
    if s == "tlm":
        return "clm_full"  # input perplexity: CLM on prompt only, no gold / no gen labels
    if prompt_only_tta:
        return "clm_full"
    return "clm_response"


def strategy_suppresses_response(strategy_name: str) -> bool:
    """若为 True，TTA 批构建时不应拼入 ``response`` 字符串（仅 TLM 需要）。"""
    s = str(strategy_name or "tent").lower()
    return s == "tlm"


def build_tta_batch(
    adapter: ModelAdapter,
    *,
    chat_prompt_text: str,
    prompt_plain: str = "",
    images: Optional[List[Any]] = None,
    messages: Optional[List[Any]] = None,
    response: Optional[str] = None,
    max_length: int = 2048,
    device: Optional[torch.device] = None,
    label_mode: str = "none",
    enable_thinking: bool = True,
    mm_encode_like_inference: bool = False,
) -> dict[str, Any]:
    """Single entry for TTA batches: ERQA, MMLU, EmbodiedBench hooks.

    ``label_mode``: ``none`` | ``clm_response`` | ``clm_full``.

    ``messages``: when provided (with actual PIL images embedded in content),
    ``build_forward_inputs`` will re-apply the chat template to obtain a fresh
    prompt text whose ``<|image_pad|>`` count matches the vision encoder output.

    ``mm_encode_like_inference``: pass True when ``chat_prompt_text`` is exactly
    the string used for ``generate`` (fair TTA vs inference on ERQA / SGLang).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return adapter.build_forward_inputs(
        chat_prompt_text=chat_prompt_text,
        prompt_plain=prompt_plain,
        images=images,
        messages=messages,
        response=response,
        max_length=max_length,
        device=device,
        label_mode=label_mode,
        enable_thinking=enable_thinking,
        mm_encode_like_inference=mm_encode_like_inference,
    )
