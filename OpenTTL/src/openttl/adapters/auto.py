from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from openttl.adapters.base import ModelAdapter
from openttl.adapters.registry import register
from openttl.models.loader import load_causal_lm, load_tokenizer


def _dtype_from_string(name: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower().replace("torch.", "")
    if key not in m:
        raise ValueError(f"Unknown torch_dtype: {name}")
    return m[key]


def _model_kwargs_from_cfg(mc: Any) -> dict:
    dtype = _dtype_from_string(str(getattr(mc, "torch_dtype", "bfloat16")))
    attn = getattr(mc, "attn_implementation", None)
    if getattr(mc, "use_flash_attention_2", False):
        attn = "flash_attention_2"
    kwargs = dict(
        pretrained_model_name_or_path=mc.pretrained_model_name_or_path,
        revision=getattr(mc, "revision", None),
        trust_remote_code=bool(getattr(mc, "trust_remote_code", True)),
        torch_dtype=dtype,
        device_map=getattr(mc, "device_map", None),
    )
    if attn:
        kwargs["attn_implementation"] = attn
    return kwargs


@register("auto")
class AutoMultimodalAdapter(ModelAdapter):
    """Default: AutoProcessor + ImageTextToText with CausalLM fallback; tokenizer-only path if no processor."""

    def __init__(self) -> None:
        self._processor: Any = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._supports_vision: bool = False

    @property
    def supports_vision(self) -> bool:
        return self._supports_vision

    def load_model(self, cfg: Any) -> PreTrainedModel:
        mc = cfg if hasattr(cfg, "pretrained_model_name_or_path") else cfg
        kwargs = _model_kwargs_from_cfg(mc)
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(**kwargs)
        except (ValueError, KeyError, OSError, TypeError):
            pass
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(**kwargs)
        except (ValueError, KeyError, OSError, TypeError, ImportError):
            pass
        return load_causal_lm(mc)

    def load_processor(self, cfg: Any) -> Any:
        mc = cfg if hasattr(cfg, "pretrained_model_name_or_path") else cfg
        path = mc.pretrained_model_name_or_path
        trust = bool(getattr(mc, "trust_remote_code", True))
        rev = getattr(mc, "revision", None)

        self._processor = None
        try:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                path,
                revision=rev,
                trust_remote_code=trust,
            )
        except Exception:
            self._processor = None

        self._tokenizer = load_tokenizer(mc)
        if self._processor is not None:
            self._tokenizer = getattr(self._processor, "tokenizer", self._tokenizer)

        self._supports_vision = bool(
            self._processor is not None 
            and hasattr(self._processor, "image_processor")
        )
        return self._processor if self._processor is not None else self._tokenizer

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> Any:
        proc = self._processor
        if proc is None:
            raise RuntimeError("apply_chat_template requires AutoProcessor; load_processor first.")
        fn = getattr(proc, "apply_chat_template", None)
        if fn is None:
            raise RuntimeError("Processor has no apply_chat_template")
        kwargs: dict[str, Any] = dict(
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        try:
            sig = inspect.signature(fn)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = enable_thinking
        except (TypeError, ValueError):
            pass
        return fn(messages, **kwargs)

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
        """label_mode: ``none`` | ``clm_response`` | ``clm_full``.

        ``messages``: when provided with actual PIL images embedded in content
        (``{"type": "image", "image": pil_img}``), the chat template is re-applied
        here so that the number of ``<|image_pad|>`` tokens in the tokenized text
        is computed from the real image dimensions, matching the vision encoder
        output in ``pixel_values`` and preventing shape-mismatch errors.
        Per-image pad count matches HF Qwen2/3-VL: ``grid.prod() // merge_size**2``.

        ``mm_encode_like_inference``: use ``chat_prompt_text`` (same as generation)
        with ``processor(text=[...], images=...)``; skips the two-step / no-gen
        template path so TTA sees the same prompt string as SGLang inference.
        """
        tok = self.tokenizer()
        proc = self._processor
        use_mm = bool(images) and self._supports_vision and proc is not None

        if response is None:
            response = ""

        if use_mm:
            prompt_enc: dict[str, Any]
            used_inference_mm = False
            if mm_encode_like_inference and str(chat_prompt_text).strip():
                try:
                    _pe_try = proc(
                        text=[chat_prompt_text],
                        images=images,
                        return_tensors="pt",
                    )
                    if _pe_try and "input_ids" in _pe_try:
                        prompt_enc = _pe_try
                        used_inference_mm = True
                except Exception:
                    used_inference_mm = False

            if not used_inference_mm:
                # ── Multimodal: token count per image = grid.prod() // merge_size**2
                # (Qwen2/3-VL; see HuggingFace processing_qwen2_vl / processing_qwen3_vl).
                # Pixel/ grid from the image processor are the single source of truth;
                # chat template is applied for text, then each <|image_pad|> is expanded
                # like the official processor (one template token per image, in order).
                _img_proc = getattr(proc, "image_processor", None)
                _used_two_step = False
                if messages is not None and _img_proc is not None:
                    try:
                        _img_data = _img_proc(images, return_tensors="pt")
                        _grid_thw = _img_data.get("image_grid_thw")
                        if _grid_thw is not None and _grid_thw.shape[0] > 0:
                            _merge_size = int(getattr(_img_proc, "merge_size", 2))
                            _merge_length = _merge_size**2
                            _image_pad_str = getattr(proc, "image_token", "<|image_pad|>")
                            _ph = "<|placeholder|>"
                            _fresh_text = self.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                                enable_thinking=enable_thinking,
                            )
                            _n_img = int(_grid_thw.shape[0])
                            _idx = 0
                            while _image_pad_str in _fresh_text:
                                if _idx >= _n_img:
                                    raise RuntimeError(
                                        "more image_token placeholders in template than images in batch"
                                    )
                                g = _grid_thw[_idx]
                                num_i = int(g.prod().item() // _merge_length)
                                _fresh_text = _fresh_text.replace(
                                    _image_pad_str, _ph * num_i, 1
                                )
                                _idx += 1
                            _fresh_text = _fresh_text.replace(_ph, _image_pad_str)
                            # Templates with only empty <|vision_start|><|vision_end|> pairs
                            while _idx < _n_img and re.search(
                                r"<\|vision_start\|>\s*<\|vision_end\|>", _fresh_text
                            ):
                                g = _grid_thw[_idx]
                                num_i = int(g.prod().item() // _merge_length)
                                _pads = _image_pad_str * num_i
                                _fresh_text = re.sub(
                                    r"(<\|vision_start\|>)\s*(<\|vision_end\|>)",
                                    r"\1" + _pads + r"\2",
                                    _fresh_text,
                                    count=1,
                                )
                                _idx += 1
                            if _idx < _n_img:
                                raise RuntimeError(
                                    "not all images could be expanded into the chat template; "
                                    "use processor path"
                                )
                            _proc_tok = getattr(proc, "tokenizer", tok)
                            _tok_enc = _proc_tok(
                                _fresh_text,
                                return_tensors="pt",
                                add_special_tokens=False,
                            )
                            prompt_enc = {
                                "input_ids": _tok_enc["input_ids"],
                                "attention_mask": _tok_enc.get("attention_mask"),
                                "pixel_values": _img_data["pixel_values"],
                                "image_grid_thw": _grid_thw,
                            }
                            _used_two_step = True
                    except Exception:
                        pass  # fall through to the standard single-call path

                if not _used_two_step:
                    # Standard path: let proc handle both text and images together.
                    _prompt_for_enc = chat_prompt_text
                    if messages is not None:
                        try:
                            _prompt_for_enc = self.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                                enable_thinking=enable_thinking,
                            )
                        except Exception:
                            pass
                    prompt_enc = proc(
                        text=[_prompt_for_enc],
                        images=images,
                        return_tensors="pt",
                    )
        else:
            prompt_enc = tok(
                chat_prompt_text,
                return_tensors="pt",
                add_special_tokens=True,
            )

        p_ids = prompt_enc["input_ids"]
        if response:
            resp_ids = tok.encode(response, add_special_tokens=False, return_tensors="pt")
            full_ids = torch.cat([p_ids, resp_ids], dim=1)
        else:
            full_ids = p_ids

        _img_tok_id: Optional[int] = None
        if use_mm and proc is not None:
            _img_tok_id = getattr(proc, "image_token_id", None)
            if _img_tok_id is None and hasattr(tok, "convert_tokens_to_ids"):
                _img_tok_id = int(
                    tok.convert_tokens_to_ids(
                        str(getattr(proc, "image_token", "<|image_pad|>"))
                    )
                )

        if full_ids.shape[1] > max_length:
            if use_mm:
                # 若 prompt 前部有非图像 token，固定 [:max_length] 会在「图像块中间」截断，
                # 导致 image_token 个数 < vision 特征数。必须至少保留到最后一个 image_token。
                actual = min(full_ids.shape[1], max_length)
                if _img_tok_id is not None:
                    pos = (full_ids[0] == _img_tok_id).nonzero()
                    if pos.numel() > 0:
                        req_len = int(pos[-1].item()) + 1
                        needed = max(max_length, req_len)
                        actual = min(full_ids.shape[1], needed)
                        if req_len > max_length:
                            warnings.warn(
                                f"Multimodal prompt needs {req_len} tokens to keep all "
                                f"image_token positions (online.max_length={max_length}); "
                                f"using length {actual}. Set online.max_length >= {req_len} to "
                                f"avoid warnings and allow more text after images.",
                                UserWarning,
                                stacklevel=2,
                            )
                full_ids = full_ids[:, :actual]
                pl = min(p_ids.shape[1], actual)
            else:
                # 纯文本：从左侧截断，保留更近的上下文
                full_ids = full_ids[:, -max_length:]
                pl = min(p_ids.shape[1], full_ids.shape[1])
        else:
            pl = p_ids.shape[1] if response else full_ids.shape[1]

        attn = torch.ones_like(full_ids)
        batch: dict[str, Any] = {
            "input_ids": full_ids,
            "attention_mask": attn,
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in prompt_enc and prompt_enc[key] is not None:
                batch[key] = prompt_enc[key]

        # mm_token_type_ids（0=text, 1=image）必须与 full_ids 等长：
        # 拼接 response 后补 text 类型(0)，截断时同步截断，
        # 否则 Qwen3.5 get_rope_index 里 mask 与 token_type 长度不一致报错。
        mtt = prompt_enc.get("mm_token_type_ids")
        if mtt is not None:
            if mtt.shape[1] < full_ids.shape[1]:
                mtt = torch.cat(
                    [
                        mtt,
                        torch.zeros(
                            (mtt.shape[0], full_ids.shape[1] - mtt.shape[1]),
                            dtype=mtt.dtype,
                            device=mtt.device,
                        ),
                    ],
                    dim=1,
                )
            batch["mm_token_type_ids"] = mtt[:, : full_ids.shape[1]]

        if label_mode == "none":
            batch["labels"] = None
        elif label_mode == "clm_response":
            labels = full_ids.clone()
            if response:
                labels[:, :pl] = -100
            batch["labels"] = labels
        elif label_mode == "clm_full":
            batch["labels"] = full_ids.clone()
        else:
            raise ValueError(f"Unknown label_mode: {label_mode!r}")

        out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return out

    def build_generate_inputs(
        self,
        *,
        prompt_text: str,
        images: Optional[List[Any]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        proc = self._processor
        tok = self.tokenizer()
        use_mm = bool(images) and self._supports_vision and proc is not None

        if use_mm:
            enc = proc(
                text=[prompt_text],
                images=images,
                return_tensors="pt",
            )
        else:
            enc = tok(prompt_text, return_tensors="pt", add_special_tokens=True)

        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}

    def decode_new_tokens(self, generated_ids: torch.Tensor, input_len: int) -> str:
        proc = self._processor
        trimmed = generated_ids[:, input_len:]
        if proc is not None and hasattr(proc, "batch_decode"):
            return proc.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        tok = self.tokenizer()
        return tok.decode(trimmed[0], skip_special_tokens=True).strip()
