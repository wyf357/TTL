from __future__ import annotations

from typing import Any, Mapping, Sequence

from openttl.adapters.auto import AutoMultimodalAdapter
from openttl.adapters.registry import register


@register("qwen3_5")
class Qwen3_5Adapter(AutoMultimodalAdapter):
    """Qwen3.5 native multimodal: same as auto; explicit registry key for ``model.adapter=qwen3_5``."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> Any:
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
