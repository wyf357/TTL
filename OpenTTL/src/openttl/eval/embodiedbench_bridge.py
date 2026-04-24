"""与 EmbodiedBench 官方评测逻辑对齐的薄桥接（不 fork 其 Evaluator 实现）。"""

from __future__ import annotations

import importlib
import json
import os
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

_remote_model_local_patch_applied: bool = False


def _local_model_requires_transformers_backend_impl(model_name: str) -> bool:
    """Qwen3.5（config: model_type=qwen3_5 / Qwen3_5*）结构与 LMDeploy 当前模块映射不一致，本地改用 transformers。"""
    root = Path(str(model_name))
    if root.is_dir() and (root / "config.json").is_file():
        try:
            with open(root / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            mt = str(cfg.get("model_type", "")).lower()
            arch0 = str((cfg.get("architectures") or [""])[0])
            if mt == "qwen3_5" or "Qwen3_5" in arch0:
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    low = str(model_name).lower()
    if "qwen3.5" in low or "qwen3_5" in low or "qwen3-5" in low:
        return True
    return False


def _local_model_prefers_turbomind_impl(model_name: str) -> bool:
    """判断本地/HF id 是否为 Qwen3.5 系：此类模型用 PyTorchEngine 常依赖 fla/triton，改用 TurboMind。"""
    if _local_model_requires_transformers_backend_impl(model_name):
        return False
    root = Path(str(model_name))
    if root.is_dir() and (root / "config.json").is_file():
        try:
            with open(root / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            mt = str(cfg.get("model_type", "")).lower()
            arch0 = str((cfg.get("architectures") or [""])[0])
            if "qwen3_5" in mt or "Qwen3_5" in arch0:
                return True
            if "qwen3" in mt and ("vl" in mt or "vision" in mt):
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    low = str(model_name).lower()
    if "qwen3.5" in low or "qwen3_5" in low or "qwen3-5" in low:
        return True
    return False


def _patch_embodiedbench_remote_model_qwen35_local() -> None:
    """本地 Qwen3.5：qwen3_5 架构由 transformers 推理；其余 Qwen3 VL 仍优先 TurboMind pipeline。"""
    global _remote_model_local_patch_applied
    if _remote_model_local_patch_applied:
        return
    from embodiedbench.planner import remote_model as rm
    from lmdeploy import TurbomindEngineConfig

    if not hasattr(rm, "_local_model_prefers_turbomind"):
        rm._local_model_prefers_turbomind = _local_model_prefers_turbomind_impl  # type: ignore[attr-defined]
    if not hasattr(rm, "_local_model_requires_transformers_backend"):
        rm._local_model_requires_transformers_backend = _local_model_requires_transformers_backend_impl  # type: ignore[attr-defined]

    if getattr(rm.RemoteModel, "_openttl_qwen35_turbomind_patched", False):
        _remote_model_local_patch_applied = True
        return

    _orig_init = rm.RemoteModel.__init__

    def _init(
        self: Any,
        model_name: str,
        model_type: str = "remote",
        language_only: bool = False,
        tp: int = 1,
        task_type: Any = None,
    ) -> None:
        if model_type == "local" and rm._local_model_requires_transformers_backend(model_name):
            if os.environ.get("OPENTTL_LOCAL_BACKEND", "").lower() == "transformers":
                from openttl.eval.eb_qwen35_transformers_local import build_transformers_local_pipeline

                self.model_name = model_name
                self.model_type = model_type
                self.language_only = language_only
                self.task_type = task_type
                self.model = build_transformers_local_pipeline(model_name, dtype="float16", tp=tp)
                return
            from openttl.eval.eb_shared_runtime import get_shared_inference
            from openttl.eval.eb_sglang_local import build_sglang_local_pipeline

            self.model_name = model_name
            self.model_type = model_type
            self.language_only = language_only
            self.task_type = task_type
            shared = get_shared_inference()
            self.model = build_sglang_local_pipeline(
                model_name,
                dtype="float16",
                tp=tp,
                inference_engine=shared,
            )
            return
        if model_type == "local" and rm._local_model_prefers_turbomind(model_name):
            self.model_name = model_name
            self.model_type = model_type
            self.language_only = language_only
            self.task_type = task_type
            backend = TurbomindEngineConfig(session_len=12000, dtype="float16", tp=tp)
            self.model = rm.pipeline(self.model_name, backend_config=backend)
            return
        _orig_init(self, model_name, model_type, language_only, tp, task_type)

    rm.RemoteModel.__init__ = _init  # type: ignore[method-assign]
    setattr(rm.RemoteModel, "_openttl_qwen35_turbomind_patched", True)
    _remote_model_local_patch_applied = True


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

    init = getattr(embodiedbench, "__file__", None)
    if init:
        return Path(init).resolve().parent
    # 上游仓库常为无 __init__.py 的 namespace 包，此时 __file__ 为 None
    paths = getattr(embodiedbench, "__path__", None)
    if paths:
        return Path(next(iter(paths))).resolve()
    raise RuntimeError("无法解析 embodiedbench 包路径")


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


def _patch_ai2thor_skip_prune() -> None:
    """避免 ai2thor 默认 prune_releases 删掉旧构建导致反复重新下载（设 AI2THOR_SKIP_PRUNE=0 恢复上游行为）。"""
    if os.environ.get("AI2THOR_SKIP_PRUNE", "1") != "1":
        return
    try:
        import ai2thor.controller as ac

        if getattr(ac.Controller, "_openttl_skip_prune", False):
            return

        def _noop_prune(self: Any) -> None:
            return None

        ac.Controller.prune_releases = _noop_prune  # type: ignore[method-assign]
        setattr(ac.Controller, "_openttl_skip_prune", True)
    except Exception:
        pass


def run_embodiedbench_eval(
    env_name: str,
    merged_config: dict[str, Any],
    hooks: Optional[Any] = None,
) -> None:
    """构造官方 Evaluator 并执行 ``check_config_valid`` + ``evaluate_main``。"""
    _require_embodiedbench()
    _patch_ai2thor_skip_prune()
    _patch_embodiedbench_remote_model_qwen35_local()
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
    from openttl.eval.eb_shared_runtime import clear_shared_inference

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("eval_embodiedbench 配置应为字典结构")
    env_name = container.get("eb_env")
    if not env_name:
        raise ValueError("必须在配置中设置 eb_env（如 eb-alf / eb-hab / eb-nav / eb-man）")
    env_name = str(env_name)
    merged = build_embodiedbench_merged_config(env_name, container)
    hooks = build_hooks_from_cfg(cfg)
    try:
        run_embodiedbench_eval(env_name, merged, hooks=hooks)
    finally:
        clear_shared_inference()


def build_hooks_from_cfg(cfg: DictConfig) -> Optional[Any]:
    """根据 ``tta.enabled`` / ``tta.backend`` 构造钩子；未启用则返回 None。"""
    import torch

    tta = OmegaConf.select(cfg, "tta")
    if not tta or not bool(OmegaConf.select(tta, "enabled") or False):
        return None
    backend = str(OmegaConf.select(tta, "backend") or "none")
    if backend in ("", "none"):
        return None
    if backend != "instruction_entropy":
        raise ValueError(f"未知 tta.backend: {backend!r}")

    model_cfg = OmegaConf.select(cfg, "model")
    if model_cfg is None:
        raise ValueError("tta.backend=instruction_entropy 需要同时提供 OpenTTL 的 model: 配置组")

    if os.environ.get("OPENTTL_LOCAL_BACKEND", "").lower() == "transformers":
        from openttl.eval.eb_instruction_tta_hook import LegacyInstructionEntropyTTAHook

        return LegacyInstructionEntropyTTAHook(model_cfg=model_cfg, tta_cfg=tta)

    from peft import PeftModel

    from openttl.eval.eb_instruction_tta_hook import InstructionEntropyTTAHook
    from openttl.eval.eb_shared_runtime import set_shared_inference
    from openttl.inference.sglang_engine import build_sglang_engine_from_omegaconf
    from openttl.models.loader import load_adapter
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.tta_runner import OnlineTTARunner

    runner_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if OmegaConf.select(tta, "lr") is not None:
        runner_cfg.online.lr = float(tta.lr)
    if OmegaConf.select(tta, "max_length") is not None:
        runner_cfg.online.max_length = int(tta.max_length)
    if OmegaConf.select(tta, "max_instruction_chars") is not None:
        runner_cfg.online.max_instruction_chars = int(tta.max_instruction_chars)
    runner_cfg.online.enabled = True
    if OmegaConf.select(runner_cfg, "strategy.name") is None:
        runner_cfg.strategy = OmegaConf.merge(
            OmegaConf.create({"name": "tent"}),
            OmegaConf.select(runner_cfg, "strategy") or OmegaConf.create(),
        )

    infer_backend = str(OmegaConf.select(runner_cfg, "inference.backend") or "sglang").lower()
    if infer_backend != "sglang":
        raise ValueError(
            "instruction_entropy（SGLang 对齐）需要 inference.backend=sglang；"
            "或设 OPENTTL_LOCAL_BACKEND=transformers 使用旧版钩子"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = load_adapter(runner_cfg)
    adapter.load_processor(model_cfg)
    tokenizer = adapter.tokenizer()
    base = adapter.load_model(model_cfg)
    ap = OmegaConf.select(model_cfg, "adapter_path")
    if ap:
        train_model = PeftModel.from_pretrained(base, str(ap))
    else:
        if not bool(OmegaConf.select(model_cfg, "peft.enabled") or False):
            raise ValueError("EmbodiedBench TTA + SGLang 需要 model.peft.enabled 或 adapter_path")
        train_model = inject_lora(base, model_cfg.peft)
    train_model.to(device)

    initial_path = str(OnlineTTARunner.initial_adapter_path(runner_cfg, train_model, runner_cfg.inference))
    infer = build_sglang_engine_from_omegaconf(model_cfg, runner_cfg.inference, tokenizer, initial_path)
    set_shared_inference(infer)

    runner = OnlineTTARunner(
        runner_cfg,
        model=train_model,
        adapter=adapter,
        inference=infer,
        device=device,
    )
    return InstructionEntropyTTAHook(runner)
