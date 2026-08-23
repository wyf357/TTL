from __future__ import annotations

import abc
from typing import Any, List, Mapping, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelAdapter(abc.ABC):
    """Unified interface for HF model families (text / native multimodal).

    ``supports_vision`` is runtime-detected from the loaded processor, not from YAML flags.
    """

    @property
    @abc.abstractmethod
    def supports_vision(self) -> bool:
        ...

    @abc.abstractmethod
    def load_model(self, cfg: Any) -> PreTrainedModel:
        ...

    @abc.abstractmethod
    def load_processor(self, cfg: Any) -> Any:
        """Return ``AutoProcessor`` (preferred) or ``PreTrainedTokenizer``."""

    def tokenizer(self) -> PreTrainedTokenizerBase:
        proc = getattr(self, "_processor", None)
        if proc is not None:
            tok = getattr(proc, "tokenizer", None)
            if tok is not None:
                return tok
        tok = getattr(self, "_tokenizer", None)
        if tok is None:
            raise RuntimeError("Adapter has no tokenizer; call load_processor first.")
        return tok

    @abc.abstractmethod
    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> Any:
        ...

    @abc.abstractmethod
    def build_forward_inputs(
        self,
        *,
        chat_prompt_text: str,
        prompt_plain: str,
        images: Optional[List[Any]],
        messages: Optional[List[Any]] = None,
        response: Optional[str],
        max_length: int,
        device: torch.device,
        label_mode: str,
        enable_thinking: bool = True,
        mm_encode_like_inference: bool = False,
    ) -> dict[str, Any]:
        """Batch for ``Strategy.compute_loss`` (may include ``labels`` or ``labels=None``).

        ``messages``: optional full conversation structure with actual PIL images
        embedded (``{"type": "image", "image": pil_img}``).  When provided, the
        implementation should use it to rebuild the prompt text so that image pad
        token counts match the vision encoder output.
        ``enable_thinking``: passed to :meth:`apply_chat_template` (match inference).
        ``mm_encode_like_inference``: if True, multimodal encoding uses the same
        ``chat_prompt_text`` string as generation (e.g. ``add_generation_prompt=True``)
        plus ``processor(text=[...], images=...)``, instead of re-applying the template
        with ``add_generation_prompt=False``.
        """

    @abc.abstractmethod
    def build_generate_inputs(
        self,
        *,
        prompt_text: str,
        images: Optional[List[Any]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        ...

    @abc.abstractmethod
    def decode_new_tokens(self, generated_ids: torch.Tensor, input_len: int) -> str:
        ...

    def score_logprob_sum(
        self,
        model: torch.nn.Module,
        *,
        full_text: str,
        prefix_len_tokens: int,
        device: torch.device,
    ) -> float:
        """Default text-only logprob sum (MMLU-style). Multimodal overrides may use images."""
        import torch.nn.functional as F

        tok = self.tokenizer()
        enc = tok(full_text, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        full_ids = enc["input_ids"]
        if full_ids.shape[1] <= prefix_len_tokens:
            return float("-inf")
        attn = enc.get("attention_mask")
        model.eval()
        with torch.no_grad():
            out = model(**{**enc, "attention_mask": attn} if attn is not None else enc)
            logits = out.logits[0].float()
            logp = F.log_softmax(logits, dim=-1)
        total = 0.0
        for t in range(prefix_len_tokens - 1, full_ids.shape[1] - 1):
            tid = int(full_ids[0, t + 1].item())
            total += float(logp[t, tid].item())
        return total
