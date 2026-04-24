
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# 注意：不要在此处 from openttl.adapters.registry 导入。registry 在导入时会
# _ensure_registered() → auto → 回到本模块，若顶层已依赖 registry 会形成循环导入。


@dataclass
class ModelLoadConfig:
    pretrained_model_name_or_path: str
    revision: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "sdpa"
    device_map: Optional[Any] = None
    use_flash_attention_2: bool = False


def extract_model_cfg(cfg: Any) -> Any:
    """自 Hydra / 训练配置中取出 ``model`` 子配置（实现位于 registry，延迟导入避免循环）。"""
    from openttl.adapters.registry import extract_model_cfg as _impl

    return _impl(cfg)


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


def load_tokenizer(
    cfg: Any,
) -> PreTrainedTokenizerBase:
    tok_cfg = getattr(cfg, "tokenizer", None) or {}
    tok = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        revision=getattr(cfg, "revision", None),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", True)),
        padding_side=getattr(tok_cfg, "padding_side", "right"),
    )
    ml = getattr(tok_cfg, "model_max_length", None)
    if ml is not None:
        tok.model_max_length = int(ml)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(cfg: Any) -> PreTrainedModel:
    mc = cfg if isinstance(cfg, ModelLoadConfig) else cfg
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
    return AutoModelForCausalLM.from_pretrained(**kwargs)


def load_adapter(cfg: Any):
    """Resolve :class:`~openttl.adapters.base.ModelAdapter` from Hydra ``model`` config."""
    from openttl.adapters.registry import resolve_adapter

    return resolve_adapter(cfg)


def load_model_for_tta(cfg: Any) -> PreTrainedModel:
    """Load trainable model for TTA (native multimodal or causal LM via adapter)."""
    mc = extract_model_cfg(cfg)
    return load_adapter(cfg).load_model(mc)


def load_causal_lm_eval(cfg: Any) -> PreTrainedModel:
    model = load_causal_lm(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_model_for_tta_eval(cfg: Any) -> PreTrainedModel:
    model = load_model_for_tta(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
