
from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn


class Strategy(abc.ABC):
    """TTA 策略：在 Trainer.compute_loss 中调用。"""

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def setup(self, model: nn.Module, teacher_model: Optional[nn.Module] = None) -> None:
        self.teacher_model = teacher_model

    @abc.abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        raise NotImplementedError

    def on_batch_end(self, trainer: Any, batch: Dict[str, torch.Tensor], loss: torch.Tensor) -> None:
        pass

    def select_samples(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        per_sample_loss: torch.Tensor,
    ) -> torch.Tensor:
        """返回 (B,) bool，True 表示参与 loss。"""
        return torch.ones(per_sample_loss.shape[0], dtype=torch.bool, device=per_sample_loss.device)
