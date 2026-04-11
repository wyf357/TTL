"""与 EmbodiedBench 官方评测逻辑对齐的薄桥接（不 fork 其 Evaluator 实现）。"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from openttl.eval.eb_runtime_hooks import (
    NoOpEmbodiedBenchHooks,
    embodiedbench_hooks,
    ensure_embodiedbench_env_patches,
    patch_evaluator_save_episode_metric,
)

_EVAL_CLASS_NAMES: dict[str, str] = {
    "eb-alf": "EB_AlfredEvaluator",
    "eb-hab": "EB_HabitatEvaluator",
    "eb-nav": "EB_NavigationEvaluator",
    "eb-man": "EB_ManipulationEvaluator",
}

_EVAL_MODULE_NAMES: dict[str, str] = {
    "eb-alf": "eb_alfred_evaluator",
    "eb-hab": "eb_habitat_evaluator",
    "eb-nav": "eb_navigation_evaluator",
    "eb-man": "eb_manipulation_evaluator",
}


def _require_embodiedbench() -> None:
    try:
        import embodiedbench  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "未安装 EmbodiedBench。请在远程环境中执行 "
            '`pip install -e ".[embodiedbench]"` 或先 `pip install -e /path/to/EmbodiedBench`，'
            "并保证仿真与数据已按上游 README 配置。"
        ) from e


def embodiedbench_package_dir() -> Path:
    _require_embodiedbench()
    import embodiedbench

    return Path(embodiedbench.__file__).resolve().parent


def get_embodiedbench_evaluator_class(env_name: str) -> type:
    """与 ``embodiedbench.main.get_evaluator`` 等价，但不 import ``embodiedbench.main``（避免其顶层副作用）。"""
    _require_embodiedbench()
    if env_name not in _EVAL_MODULE_NAMES:
        raise ValueError(f"未知 EmbodiedBench 环境: {env_name!r}")
    mod_name = f"embodiedbench.evaluator.{_EVAL_MODULE_NAMES[env_name]}"
    cls_name = _EVAL_CLASS_NAMES[env_name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def build_embodiedbench_merged_config(
    env_name: str, overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """读取 ``embodiedbench/configs/{env_name}.yaml`` 并与 overrides 合并（与官方 main 一致）。"""
    yaml_path = embodiedbench_package_dir() / "configs" / f"{env_name}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"找不到 EmbodiedBench 配置: {yaml_path}")
    base_cfg = OmegaConf.load(str(yaml_path))
    base = OmegaConf.to_container(base_cfg, resolve=True)
    if not isinstance(base, dict):
        raise TypeError(f"EmbodiedBench 基础配置应为 dict: {yaml_path}")
    merged = {**base}
    skip = frozenset({"eb_env", "tta", "defaults", "_target_", "model"})
    for k, v in dict(overrides).items():
        ks = str(k)
        if ks.startswith("hydra") or k in skip:
            continue
        if v is not None and k != "env":
            merged[k] = v
    return merged


def _strip_for_evaluator(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        ks = str(k)
        if ks.startswith("hydra") or k in ("eb_env", "tta", "defaults", "_target_", "model"):
            continue
        if k == "env":
            continue
        out[k] = v
    return out


def run_embodiedbench_eval(
    env_name: str,
    merged_config: dict[str, Any],
    hooks: Optional[Any] = None,
) -> None:
    """构造官方 Evaluator 并执行 ``check_config_valid`` + ``evaluate_main``。"""
    _require_embodiedbench()
    ensure_embodiedbench_env_patches()
    ev_cfg = _strip_for_evaluator(dict(merged_config))
    cls = get_embodiedbench_evaluator_class(env_name)
    evaluator = cls(ev_cfg)
    patch_evaluator_save_episode_metric(evaluator, hooks)
    h = hooks if hooks is not None else NoOpEmbodiedBenchHooks()
    with embodiedbench_hooks(h):
        evaluator.check_config_valid()
        evaluator.evaluate_main()


def run_embodiedbench_from_omegaconf(cfg: DictConfig) -> None:
    """从 OpenTTL Hydra 配置运行（顶层 ``eb_env`` + 与官方一致的字段）。"""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("eval_embodiedbench 配置应为字典结构")
    env_name = container.get("eb_env")
    if not env_name:
        raise ValueError("必须在配置中设置 eb_env（如 eb-alf / eb-hab / eb-nav / eb-man）")
    env_name = str(env_name)
    merged = build_embodiedbench_merged_config(env_name, container)
    hooks = build_hooks_from_cfg(cfg)
    run_embodiedbench_eval(env_name, merged, hooks=hooks)


def build_hooks_from_cfg(cfg: DictConfig) -> Optional[Any]:
    """根据 ``tta.enabled`` / ``tta.backend`` 构造钩子；未启用则返回 None。"""
    tta = OmegaConf.select(cfg, "tta")
    if not tta or not bool(OmegaConf.select(tta, "enabled") or False):
        return None
    backend = str(OmegaConf.select(tta, "backend") or "none")
    if backend in ("", "none"):
        return None
    if backend == "instruction_entropy":
        from openttl.eval.eb_instruction_tta_hook import InstructionEntropyTTAHook

        model_cfg = OmegaConf.select(cfg, "model")
        if model_cfg is None:
            raise ValueError("tta.backend=instruction_entropy 需要同时提供 OpenTTL 的 model: 配置组")
        return InstructionEntropyTTAHook(model_cfg=model_cfg, tta_cfg=tta)
    raise ValueError(f"未知 tta.backend: {backend!r}")
