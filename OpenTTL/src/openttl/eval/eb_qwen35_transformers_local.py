"""EmbodiedBench local 推理：Qwen3.5（model_type=qwen3_5）在 LMDeploy 0.12.x 下易触发配置/架构不兼容，改用 transformers 原生 VLM 推理。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Union

import torch


def _openai_messages_to_qwen3_vl_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """EmbodiedBench 使用 OpenAI 风格（image_url）；Qwen3VLProcessor 期望 type=image + image=url/path。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        new_content: List[Dict[str, Any]] = []
        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})
        else:
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    new_content.append({"type": "text", "text": part["text"]})
                elif ptype == "image_url":
                    url = part["image_url"]["url"]
                    new_content.append({"type": "image", "image": url})
                else:
                    raise ValueError(f"Unsupported message content type: {ptype!r}")
        out.append({"role": role, "content": new_content})
    return out


@dataclass
class _TextResponse:
    text: str


class EmbodiedBenchTransformersLocalPipeline:
    """与 lmdeploy pipeline 对齐的最小接口：``__call__(messages, gen_config=...) -> .text``。"""

    def __init__(self, model_path: str, dtype: str = "float16", tp: int = 1) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._tp = int(tp)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # 默认固定到「当前进程可见的第 N 张 CUDA 卡」（OPENTTL_INFERENCE_CUDA_DEVICE，与 TTA_GPU 对齐），
        # 避免 device_map="auto" 在单卡能装下时仍只占用 cuda:0，却与「渲染用另一张卡」的预期混淆；
        # 需要多卡自动切分时设 OPENTTL_DEVICE_MAP=auto。
        _dm = os.environ.get("OPENTTL_DEVICE_MAP", "").strip().lower()
        if _dm == "auto":
            device_map: Union[str, Dict[str, Any]] = "auto"
        else:
            try:
                _dev = int(os.environ.get("OPENTTL_INFERENCE_CUDA_DEVICE", "0"))
            except ValueError:
                _dev = 0
            device_map = {"": _dev}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, messages: List[Mapping[str, Any]], gen_config: Any = None) -> _TextResponse:
        qwen_msgs = _openai_messages_to_qwen3_vl_messages(messages)
        inputs = self.processor.apply_chat_template(
            qwen_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        max_new_tokens = 4096
        temperature = 0.0
        if gen_config is not None:
            max_new_tokens = int(getattr(gen_config, "max_new_tokens", max_new_tokens))
            temperature = float(getattr(gen_config, "temperature", temperature))

        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens)
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            top_p = getattr(gen_config, "top_p", None)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
        else:
            gen_kwargs["do_sample"] = False

        generated = self.model.generate(**inputs, **gen_kwargs)
        in_len = inputs["input_ids"].shape[1]
        trimmed = generated[:, in_len:]
        out_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _TextResponse(text=out_text)


def build_transformers_local_pipeline(model_path: str, dtype: str = "float16", tp: int = 1) -> EmbodiedBenchTransformersLocalPipeline:
    return EmbodiedBenchTransformersLocalPipeline(model_path, dtype=dtype, tp=tp)
