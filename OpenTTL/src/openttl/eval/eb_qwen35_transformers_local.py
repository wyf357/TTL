"""EmbodiedBench local 推理：Qwen3.5（model_type=qwen3_5）在 LMDeploy 0.12.x 下易触发配置/架构不兼容，改用 transformers 原生 VLM 推理。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Union

import torch

# Qwen3.5 tokenizer added tokens（id 248068 / 248069），与错误正则里的字面量 "redacted_thinking" 无关
_QWEN35_THINK_OPEN = "<" + "think" + ">"
_QWEN35_THINK_CLOSE = "</" + "think" + ">"


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


def _qwen35_enable_thinking_from_env() -> bool:
    """Qwen3.5：`apply_chat_template(..., enable_thinking=...)` 控制是否在模板里展开思考段。

    默认关闭（与 EmbodiedBench 的纯 JSON 规划输出一致）。需要思考链时设
    ``OPENTTL_QWEN35_ENABLE_THINKING=1``。
    """
    v = os.environ.get("OPENTTL_QWEN35_ENABLE_THINKING", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return False


def _strip_qwen35_thinking_traces(text: str) -> str:
    """去掉解码文本中的 think 标签段，并尽量保留「最终回答」。

    enable_thinking=False 只影响 prompt；模型仍可能输出思考段。另：部分 checkpoint 里 think 相关
    added tokens 的 special 为 false，skip_special_tokens 仍可能留下标签字面量。

    关闭本后处理：环境变量 OPENTTL_QWEN35_STRIP_THINKING=0。
    """
    v = os.environ.get("OPENTTL_QWEN35_STRIP_THINKING", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return text
    if not text:
        return text
    o, c = _QWEN35_THINK_OPEN, _QWEN35_THINK_CLOSE
    prev = None
    while prev != text:
        prev = text
        text = re.sub(re.escape(o) + r".*?" + re.escape(c), "", text, flags=re.DOTALL)
    text = text.strip()
    # 仅有闭合标签、或「假 JSON + 闭合 + 真 JSON」：取最后一个闭合标签之后（对齐 Qwen3-Thinking 文档按 token 切分思路）
    if c in text:
        tail = text.split(c)[-1].strip()
        if tail:
            text = tail
    return text.strip()


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
            enable_thinking=_qwen35_enable_thinking_from_env(),
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
        out_text = _strip_qwen35_thinking_traces(out_text)
        return _TextResponse(text=out_text)


def build_transformers_local_pipeline(model_path: str, dtype: str = "float16", tp: int = 1) -> EmbodiedBenchTransformersLocalPipeline:
    return EmbodiedBenchTransformersLocalPipeline(model_path, dtype=dtype, tp=tp)
