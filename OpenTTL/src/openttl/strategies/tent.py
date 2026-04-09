
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
from torch import nn

from openttl.strategies.base import Strategy
from openttl.strategies.tta_shared import apply_backbone_eval_lora_train, masked_mean_sequence_entropy


class TentStrategy(Strategy):
    """TENT-LLM: LoRA + mean sequence Shannon entropy minimization (algorithms/tent.md)."""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        apply_backbone_eval_lora_train(model)
        labels = inputs.get("labels")
        fwd = {k: v for k, v in inputs.items() if k != "labels"}
        out = model(**fwd)
        logits = out.logits
        if labels is None:
            am = inputs.get("attention_mask")
            if am is not None:
                mask = am.ne(0)
            else:
                mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
        else:
            mask = labels.ne(-100)

        eps = float(getattr(self.cfg, "epsilon", 1e-8))
        loss = masked_mean_sequence_entropy(logits, mask, eps=eps)
        return (loss, out) if return_outputs else loss
