"""首期 TTA：在 ``language_only`` 设定下，每个 episode 结束后用指令文本做一次 Tent 熵最小步。

与 EmbodiedBench 内的 ``VLMPlanner`` / ``RemoteModel`` 解耦：使用 OpenTTL 的 causal LM + TentStrategy，
仅作测试时适应占位与可扩展接口；若要与仿真内 planner 共用权重，需在服务器侧自行对齐 checkpoint。"""

from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from openttl.eval.eb_runtime_hooks import EmbodiedBenchHooks
from openttl.models.loader import load_causal_lm, load_tokenizer
from openttl.strategies.tent import TentStrategy


class InstructionEntropyTTAHook(EmbodiedBenchHooks):
    def __init__(self, model_cfg: DictConfig, tta_cfg: DictConfig) -> None:
        self._model_cfg = model_cfg
        self._tta_cfg = tta_cfg
        self._model: Optional[torch.nn.Module] = None
        self._tokenizer = None
        self._strategy: Optional[TentStrategy] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        tokenizer = load_tokenizer(self._model_cfg)
        model = load_causal_lm(self._model_cfg)
        ap = OmegaConf.select(self._model_cfg, "adapter_path")
        if ap:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(ap))
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
        self._tokenizer = tokenizer
        self._model = model

    def on_episode_start(self, env: Any, reset_output: Any) -> None:
        return None

    def on_env_step_end(self, env: Any, step_output: Any) -> None:
        return None

    def on_episode_end(self, evaluator: Any, episode_info: dict) -> None:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        assert self._strategy is not None and self._optimizer is not None

        text = str(episode_info.get("instruction") or "")
        max_chars = int(OmegaConf.select(self._tta_cfg, "max_instruction_chars") or 2048)
        text = text[:max_chars]
        if not text.strip():
            return

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(OmegaConf.select(self._tta_cfg, "max_length") or 512),
            padding="max_length",
        )
        batch = {k: v.to(self._device) for k, v in enc.items()}
        batch["labels"] = None

        self._optimizer.zero_grad(set_to_none=True)
        loss = self._strategy.compute_loss(self._model, batch)
        loss.backward()
        self._optimizer.step()
