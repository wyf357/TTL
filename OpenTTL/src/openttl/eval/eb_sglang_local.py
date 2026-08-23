"""EmbodiedBench local 推理：Qwen3.5 系优先 SGLang（多模态 + 动态 LoRA）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch

from openttl.eval.eb_qwen35_transformers_local import _openai_messages_to_qwen3_vl_messages
from openttl.inference.base import InferenceEngine


@dataclass
class _TextResponse:
    text: str


def _messages_to_prompt_openai_style(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, Optional[List[Any]]]:
    """返回 (prompt_string, image_list_or_none)。仅支持 EB OpenAI 风格 + Qwen3-VL。"""
    flat_images: List[Any] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    flat_images.append(url)
    qwen_msgs = _openai_messages_to_qwen3_vl_messages(messages)
    text = processor.apply_chat_template(
        qwen_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not flat_images:
        return text, None
    return text, flat_images


class EmbodiedBenchSGLangLocalPipeline:
    """与 lmdeploy pipeline 对齐的最小接口：``__call__(messages, gen_config=...) -> .text``。"""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "float16",
        tp: int = 1,
        inference_engine: Optional[InferenceEngine] = None,
        processor: Any = None,
    ) -> None:
        self._tp = int(tp)
        self._model_path = model_path
        self._engine_wrapper: Optional[InferenceEngine] = inference_engine
        self._owns_engine = inference_engine is None

        if processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.processor = processor

        if self._engine_wrapper is None:
            from sglang import Engine

            kwargs: Dict[str, Any] = dict(
                model_path=model_path,
                dtype=dtype,
                tp_size=int(tp),
                trust_remote_code=True,
                mem_fraction_static=float(os.environ.get("OPENTTL_SGLANG_MEM_FRACTION", "0.45")),
                disable_cuda_graph=True,
                log_level="error",
            )
            self._sgl_engine = Engine(**kwargs)
        else:
            self._sgl_engine = getattr(inference_engine, "engine", None)

    @property
    def engine(self) -> Any:
        return self._sgl_engine

    @torch.inference_mode()
    def __call__(self, messages: List[Mapping[str, Any]], gen_config: Any = None) -> _TextResponse:
        prompt, images = _messages_to_prompt_openai_style(self.processor, messages)
        max_new_tokens = 4096
        temperature = 0.0
        top_p = 1.0
        if gen_config is not None:
            max_new_tokens = int(getattr(gen_config, "max_new_tokens", max_new_tokens))
            temperature = float(getattr(gen_config, "temperature", temperature))
            top_p = float(getattr(gen_config, "top_p", top_p))

        sampling: Dict[str, Any] = dict(max_new_tokens=max_new_tokens, top_p=top_p)
        if temperature > 0:
            sampling["temperature"] = temperature
        else:
            sampling["temperature"] = 0.0

        if self._engine_wrapper is not None:
            out = self._engine_wrapper.generate(
                prompt,
                image_data=images,
                sampling_params=sampling,
                lora_name=self._engine_wrapper.current_lora_name,
            )
            text = str(out).strip()
            return _TextResponse(text=text)

        assert self._sgl_engine is not None
        out = self._sgl_engine.generate(
            prompt=prompt,
            image_data=images,
            sampling_params=sampling,
        )
        if isinstance(out, dict):
            text = str(out.get("text", "")).strip()
        else:
            text = str(out).strip()
        return _TextResponse(text=text)


def build_sglang_local_pipeline(
    model_path: str,
    dtype: str = "float16",
    tp: int = 1,
    *,
    inference_engine: Optional[InferenceEngine] = None,
    processor: Any = None,
) -> EmbodiedBenchSGLangLocalPipeline:
    return EmbodiedBenchSGLangLocalPipeline(
        model_path,
        dtype=dtype,
        tp=tp,
        inference_engine=inference_engine,
        processor=processor,
    )
