from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from openttl.adapters.base import ModelAdapter
from openttl.inference.base import InferenceEngine


class HuggingFaceEngine(InferenceEngine):
    """HF ``model.generate`` with multimodal inputs via :class:`~openttl.adapters.base.ModelAdapter`."""

    def __init__(
        self,
        model: torch.nn.Module,
        adapter: ModelAdapter,
        device: torch.device,
    ) -> None:
        self._model = model
        self._adapter = adapter
        self._device = device
        self._fake_lora: Optional[str] = "hf_baseline"

    @property
    def current_lora_name(self) -> Optional[str]:
        return self._fake_lora

    def sync_lora(self, local_dir: str, new_name: str) -> str:
        del local_dir
        self._fake_lora = new_name
        return new_name

    def generate(
        self,
        prompt: Union[str, List[str]],
        *,
        image_data: Optional[Any] = None,
        sampling_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
    ) -> Union[str, List[str]]:
        del lora_name
        single = isinstance(prompt, str)
        prompts: List[str] = [prompt] if single else list(prompt)
        imgs: Optional[List[Any]] = None
        if image_data is not None:
            if isinstance(image_data, list):
                imgs = image_data
            else:
                imgs = [image_data]

        sp = sampling_params if isinstance(sampling_params, dict) else (sampling_params[0] if sampling_params else {})
        max_new = int((sp or {}).get("max_new_tokens", 256))
        temperature = float((sp or {}).get("temperature", 1.0))
        top_p = float((sp or {}).get("top_p", 1.0))
        do_sample = temperature > 0 and (sp or {}).get("do_sample", temperature > 0)

        outs: List[str] = []
        self._model.eval()
        tok = self._adapter.tokenizer()
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)

        for p in prompts:
            enc = self._adapter.build_generate_inputs(
                prompt_text=p,
                images=imgs,
                device=self._device,
            )
            with torch.no_grad():
                gkw: Dict[str, Any] = dict(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    pad_token_id=pad_id,
                )
                if do_sample:
                    gkw["temperature"] = temperature
                    gkw["top_p"] = top_p
                gen_ids = self._model.generate(**gkw)
            in_len = int(enc["input_ids"].shape[1])
            text = self._adapter.decode_new_tokens(gen_ids, in_len)
            outs.append(text.strip())
        return outs[0] if single else outs

    def score_logprob_sum(
        self,
        *,
        full_text: str,
        prefix_len_tokens: int,
        lora_name: Optional[str] = None,
    ) -> float:
        del lora_name
        return float(
            self._adapter.score_logprob_sum(
                self._model,
                full_text=full_text,
                prefix_len_tokens=prefix_len_tokens,
                device=self._device,
            )
        )
