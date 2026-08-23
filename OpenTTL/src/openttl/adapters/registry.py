from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Type

from openttl.adapters.base import ModelAdapter

_REGISTRY: Dict[str, Type[ModelAdapter]] = {}


def register(name: str) -> Callable[[Type[ModelAdapter]], Type[ModelAdapter]]:
    def deco(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
        _REGISTRY[name] = cls
        return cls

    return deco


def extract_model_cfg(cfg: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        m = OmegaConf.select(cfg, "model")
        if m is not None:
            return m
    except Exception:
        pass
    return getattr(cfg, "model", None) or cfg


def _read_hub_config(model_path: str) -> dict:
    root = Path(str(model_path))
    if root.is_dir() and (root / "config.json").is_file():
        try:
            with open(root / "config.json", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def resolve_adapter(cfg: Any) -> ModelAdapter:
    """Resolve adapter: explicit ``model.adapter`` → config.json heuristics → AutoMultimodalAdapter."""
    mc = extract_model_cfg(cfg)
    explicit: Any = None
    try:
        from omegaconf import OmegaConf

        explicit = OmegaConf.select(mc, "adapter")
    except Exception:
        explicit = getattr(mc, "adapter", None)

    if explicit is not None and str(explicit).strip() and str(explicit).lower() != "null":
        key = str(explicit).strip()
        if key not in _REGISTRY:
            raise ValueError(f"Unknown model.adapter: {key!r}; known: {sorted(_REGISTRY)}")
        return _REGISTRY[key]()

    path = str(getattr(mc, "pretrained_model_name_or_path", "") or "")
    hub = _read_hub_config(path)
    mt = str(hub.get("model_type", "") or "").lower()
    arch0 = str((hub.get("architectures") or [""])[0])

    if mt == "qwen3_5" or "Qwen3_5" in arch0:
        return _REGISTRY.get("qwen3_5", _REGISTRY["auto"])()

    return _REGISTRY["auto"]()


def load_adapter(cfg: Any) -> ModelAdapter:
    return resolve_adapter(cfg)


def _ensure_registered() -> None:
    """Import side-effect: register default and family adapters."""
    from openttl.adapters import auto as auto_mod  # noqa: F401
    from openttl.adapters import qwen3_5 as qwen_mod  # noqa: F401

    del auto_mod, qwen_mod


_ensure_registered()
