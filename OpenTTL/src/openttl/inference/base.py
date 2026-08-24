from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class InferenceEngine(ABC):
    """推理后端抽象：SGLang / HF 等。"""

    @property
    @abstractmethod
    def current_lora_name(self) -> Optional[str]:
        """当前用于 generate 的 LoRA 逻辑名（SGLang lora_path 请求体）。"""
        ...

    @abstractmethod
    def sync_lora(self, local_dir: str, new_name: str) -> str:
        """将磁盘上的 LoRA 目录加载为 new_name，并卸载旧名（若可）。返回 new_name。"""

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        *,
        image_data: Optional[Any] = None,
        sampling_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
    ) -> Union[str, List[str]]:
        ...

    @abstractmethod
    def score_logprob_sum(
        self,
        *,
        full_text: str,
        prefix_len_tokens: int,
        lora_name: Optional[str] = None,
    ) -> float:
        """对整段文本一次性前向，返回从 prefix 之后（续写部分）的 token logprob 之和。"""

    def shutdown(self) -> None:
        """释放引擎（SGLang 子进程等）。"""
        return None
