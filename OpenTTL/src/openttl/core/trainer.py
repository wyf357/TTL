
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Trainer

from openttl.strategies.base import Strategy


class TTATrainer(Trainer):
    def __init__(
        self,
        strategy: Strategy,
        teacher_model: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.tta_strategy = strategy
        self.teacher_model = teacher_model
        super().__init__(*args, **kwargs)
        self.tta_strategy.setup(self.model, teacher_model)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        del num_items_in_batch
        out = self.tta_strategy.compute_loss(model, inputs, return_outputs=return_outputs)
        if return_outputs and isinstance(out, tuple):
            loss, outputs = out
            self.tta_strategy.on_batch_end(self, inputs, loss)
            return loss, outputs
        loss = out
        self.tta_strategy.on_batch_end(self, inputs, loss)
        return loss
