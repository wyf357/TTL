from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf
from torch import nn

from openttl.adapters.base import ModelAdapter
from openttl.inference.base import InferenceEngine
from openttl.strategies import build_strategy
from openttl.strategies.base import Strategy

LOG = logging.getLogger(__name__)


def export_peft_adapter_dir(model: nn.Module, out_dir: Union[str, Path], exist_ok: bool = False) -> Path:
    """将当前 PEFT 适配器写入目录（供 SGLang ``--lora-paths`` / ``load_lora_adapter``）。"""
    del exist_ok  # 始终覆盖同名目录，保证磁盘权重与内存一致
    p = Path(out_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p))
    return p


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class OnlineTTARunner:
    """HF+PEFT 上计算 TTA loss；按步导出 LoRA 并同步到 ``InferenceEngine``。"""

    def __init__(
        self,
        cfg: Any,
        *,
        model: nn.Module,
        adapter: ModelAdapter,
        inference: InferenceEngine,
        device: torch.device,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.adapter = adapter
        self.inference = inference
        self.device = device
        self.strategy = strategy or build_strategy(cfg)

        oc = OmegaConf.select(cfg, "online") or OmegaConf.create({})
        self._lr = float(OmegaConf.select(oc, "lr") or 1e-4)
        self._wd = float(OmegaConf.select(oc, "weight_decay") or 0.0)
        # 注意不能用 `or 1`：显式配置 0（HF 后端无需同步）会被吞掉
        _se = OmegaConf.select(oc, "sync_every_n_updates")
        self._sync_every = int(_se) if _se is not None else 1
        self._adapter_root = OmegaConf.select(oc, "adapter_root")
        train_out = OmegaConf.select(cfg, "train.output_dir") or "./outputs/online_tta"
        self._adapter_root = Path(str(self._adapter_root or Path(train_out) / "online_tta_adapters"))
        self._adapter_root.mkdir(parents=True, exist_ok=True)

        if bool(OmegaConf.select(oc, "gradient_checkpointing") or False):
            # For PEFT models both calls are required:
            # 1) gradient_checkpointing_enable() activates recomputation in the base model.
            # 2) enable_input_require_grads() is needed so that LoRA adapter inputs carry
            #    gradients even though frozen base-model parameters do not require grad.
            #    Without it, gradient checkpointing silently does nothing on PEFT models.
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.strategy.setup(self.model, teacher_model=None)
        self._optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self._lr,
            weight_decay=self._wd,
        )
        self._update_idx = 0
        self._lora_step = 0

    @property
    def tokenizer(self) -> Any:
        return self.adapter.tokenizer()

    @property
    def processor(self) -> Any:
        return getattr(self.adapter, "_processor", None)

    @staticmethod
    def initial_adapter_path(cfg: Any, model: nn.Module, inference_cfg: Any) -> Path:
        """在启动 SGLang 之前调用：写出 ``tta_v0`` 权重目录。"""
        from omegaconf import OmegaConf

        oc = OmegaConf.select(cfg, "online") or OmegaConf.create({})
        train_out = OmegaConf.select(cfg, "train.output_dir") or "./outputs/online_tta"
        root = OmegaConf.select(oc, "adapter_root")
        adapter_root = Path(str(root or Path(train_out) / "online_tta_adapters"))
        adapter_root.mkdir(parents=True, exist_ok=True)
        init_name = str(OmegaConf.select(inference_cfg, "initial_lora_name") or "tta_v0")
        p = adapter_root / init_name
        export_peft_adapter_dir(model, p, exist_ok=True)
        return p

    @property
    def current_lora_name(self) -> Optional[str]:
        return self.inference.current_lora_name

    def _next_lora_name(self) -> str:
        self._lora_step += 1
        return f"tta_v{self._lora_step}"

    def update(self, batch: Dict[str, Any]) -> float:
        """单步 TTA；返回标量 loss。"""
        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)
        batch = _batch_to_device(batch, self.device)
        loss = self.strategy.compute_loss(self.model, batch)
        if getattr(self.strategy, "handles_own_backward", False):
            # 策略（如 COME 多步 rollout）已在内部逐步 backward（梯度累积），
            # 返回的是 detached 标量，这里直接取数值。
            loss_val = float(loss)
        else:
            loss.backward()
            loss_val = float(loss.detach().cpu())
        self._optimizer.step()

        # Explicitly release the computation graph and batch tensors so that the
        # CUDA allocator can reuse the memory in the next iteration.
        del loss, batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._update_idx += 1
        if self._sync_every > 0 and (self._update_idx % self._sync_every == 0):
            name = self._next_lora_name()
            out = self._adapter_root / name
            export_peft_adapter_dir(self.model, out, exist_ok=True)
            self.inference.sync_lora(str(out), name)
            LOG.info("TTA sync LoRA -> SGLang name=%s path=%s", name, out)
        return loss_val

    def enabled(self) -> bool:
        return bool(OmegaConf.select(self.cfg, "online.enabled") or False)
