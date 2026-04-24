"""EmbodiedBench 评测期间共享的推理引擎（SGLang / HF），供 RemoteModel 与 TTA Hook 对齐。"""

from __future__ import annotations

from typing import Optional

from openttl.inference.base import InferenceEngine

_shared_inference: Optional[InferenceEngine] = None


def set_shared_inference(engine: Optional[InferenceEngine]) -> None:
    global _shared_inference
    _shared_inference = engine


def get_shared_inference() -> Optional[InferenceEngine]:
    return _shared_inference


def clear_shared_inference() -> None:
    global _shared_inference
    if _shared_inference is not None and hasattr(_shared_inference, "shutdown"):
        try:
            _shared_inference.shutdown()
        except Exception:
            pass
    _shared_inference = None
