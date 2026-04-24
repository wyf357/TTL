"""TTA：episode 结束时用指令文本更新 LoRA，并通过共享 SGLang 引擎同步到仿真内 planner。

若设置环境变量 ``OPENTTL_LOCAL_BACKEND=transformers``，则回退到旧版「独立 HF 模型」路径（不与 RemoteModel 权重自动对齐）。
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from openttl.eval.eb_runtime_hooks import EmbodiedBenchHooks
from openttl.models.loader import load_adapter
from openttl.online.batching import build_tta_batch, strategy_to_label_mode
from openttl.strategies.tent import TentStrategy


def _episode_optional_images(episode_info: dict) -> Optional[List[Any]]:
    """Optional multimodal inputs from EmbodiedBench episode_info (extensible keys)."""
    if "images" in episode_info:
        v = episode_info.get("images")
        if v is None:
            return None
        return list(v) if isinstance(v, (list, tuple)) else [v]
    for k in ("image", "observation_image", "rgb", "frame"):
        if k in episode_info and episode_info[k] is not None:
            v = episode_info[k]
            return list(v) if isinstance(v, (list, tuple)) else [v]
    return None


class InstructionEntropyTTAHook(EmbodiedBenchHooks):
    """持有 :class:`openttl.online.tta_runner.OnlineTTARunner`，与共享 SGLang 推理对齐。"""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def on_episode_start(self, env: Any, reset_output: Any) -> None:
        return None

    def on_env_step_end(self, env: Any, step_output: Any) -> None:
        return None

    def on_episode_end(self, evaluator: Any, episode_info: dict) -> None:
        if not self._runner.enabled():
            return
        text = str(episode_info.get("instruction") or "")
        max_chars = int(OmegaConf.select(self._runner.cfg, "online.max_instruction_chars") or 2048)
        text = text[:max_chars]
        if not text.strip():
            return
        dev = self._runner.device
        adapter = self._runner.adapter
        max_len = int(OmegaConf.select(self._runner.cfg, "online.max_length") or 512)
        strat_name = str(OmegaConf.select(self._runner.cfg, "strategy.name") or "tent").lower()
        lm = strategy_to_label_mode(strat_name, prompt_only_tta=True)
        images = _episode_optional_images(episode_info)
        batch = build_tta_batch(
            adapter,
            chat_prompt_text=text,
            prompt_plain=text,
            images=images,
            response=None,
            max_length=max_len,
            device=dev,
            label_mode=lm,
        )
        try:
            self._runner.update(batch)
        except Exception as e:  # pragma: no cover
            import logging

            logging.getLogger(__name__).warning("TTA update failed: %s", e)


class LegacyInstructionEntropyTTAHook(EmbodiedBenchHooks):
    """旧版：独立 HF causal LM + Tent（``OPENTTL_LOCAL_BACKEND=transformers`` 时使用）。"""

    def __init__(self, model_cfg: DictConfig, tta_cfg: DictConfig) -> None:
        self._model_cfg = model_cfg
        self._tta_cfg = tta_cfg
        self._model: Optional[torch.nn.Module] = None
        self._adapter = None
        self._strategy: Optional[TentStrategy] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from peft import PeftModel

        from openttl.models.lora_wrapper import inject_lora

        adapter = load_adapter(self._model_cfg)
        adapter.load_processor(self._model_cfg)
        model = adapter.load_model(self._model_cfg)
        ap = OmegaConf.select(self._model_cfg, "adapter_path")
        if ap:
            model = PeftModel.from_pretrained(model, str(ap))
        elif bool(OmegaConf.select(self._model_cfg, "peft.enabled") or False):
            model = inject_lora(model, self._model_cfg.peft)
        model.to(self._device)
        model.train()

        lr = float(OmegaConf.select(self._tta_cfg, "lr") or 1e-4)
        self._optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )
        strat_cfg = OmegaConf.select(self._tta_cfg, "strategy")
        if strat_cfg is None:
            strat_cfg = OmegaConf.create({"epsilon": 1e-8})
        self._strategy = TentStrategy(strat_cfg)
        self._adapter = adapter
        self._model = model

    def on_episode_start(self, env: Any, reset_output: Any) -> None:
        return None

    def on_env_step_end(self, env: Any, step_output: Any) -> None:
        return None

    def on_episode_end(self, evaluator: Any, episode_info: dict) -> None:
        self._ensure_loaded()
        assert self._model is not None and self._adapter is not None
        assert self._strategy is not None and self._optimizer is not None

        text = str(episode_info.get("instruction") or "")
        max_chars = int(OmegaConf.select(self._tta_cfg, "max_instruction_chars") or 2048)
        text = text[:max_chars]
        if not text.strip():
            return

        max_len = int(OmegaConf.select(self._tta_cfg, "max_length") or 512)
        strat_name = str(OmegaConf.select(self._tta_cfg, "strategy.name") or "tent").lower()
        lm = strategy_to_label_mode(strat_name, prompt_only_tta=True)
        images = _episode_optional_images(episode_info)
        batch = build_tta_batch(
            self._adapter,
            chat_prompt_text=text,
            prompt_plain=text,
            images=images,
            response=None,
            max_length=max_len,
            device=self._device,
            label_mode=lm,
        )

        self._optimizer.zero_grad(set_to_none=True)
        loss = self._strategy.compute_loss(self._model, batch)
        loss.backward()
        self._optimizer.step()
