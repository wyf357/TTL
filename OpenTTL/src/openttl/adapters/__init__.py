"""Model adapters: unified load, processor, chat template, and multimodal batching."""

from openttl.adapters.registry import extract_model_cfg, load_adapter, resolve_adapter

__all__ = ["extract_model_cfg", "load_adapter", "resolve_adapter"]
