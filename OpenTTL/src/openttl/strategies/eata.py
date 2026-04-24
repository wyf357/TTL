
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.tent import TentStrategy
from openttl.strategies.tta_shared import (
    apply_backbone_eval_lora_train,
    masked_mean_sequence_entropy,
    tta_model_forward,
)


class EATAStrategy(TentStrategy):
    """EATA：继承 Tent 的 TTA/熵聚合约定，在熵损失前按 token 级熵过滤不可靠位置。"""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        apply_backbone_eval_lora_train(model)
        labels = inputs.get("labels")
        out = tta_model_forward(model, inputs)
        logits = out.logits
        if labels is None:
            base_mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
        else:
            base_mask = labels.ne(-100)

        p = F.softmax(logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)
        ent = -(p * logp).sum(dim=-1)

        flat = ent[base_mask]
        q = float(getattr(self.cfg, "reliable_entropy_quantile", 0.6))
        if flat.numel() == 0:
            z = logits.float().sum() * 0.0
            return (z, out) if return_outputs else z
        thresh = torch.quantile(flat, q)
        reliable = base_mask & (ent <= thresh)

        min_tok = int(getattr(self.cfg, "min_batch_tokens", 8))
        if reliable.float().sum() < min_tok:
            reliable = base_mask

        coeff = float(getattr(self.cfg, "entropy_coeff", 1.0))
        eps = float(getattr(self.cfg, "epsilon", 1e-8))
        loss = coeff * masked_mean_sequence_entropy(logits, reliable, eps=eps)
        return (loss, out) if return_outputs else loss
