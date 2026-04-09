from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.base import Strategy


def _per_sample_mean_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean NLL (nats) per sequence over valid next-token positions (CLM shift)."""
    B, T, V = logits.shape
    nll = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V),
        labels[:, 1:].reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    nll = nll.view(B, T - 1)
    valid = labels[:, 1:].ne(-100).float()
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (nll * valid).sum(dim=1) / denom


class TLMStrategy(Strategy):
    """Test-Time Learning (TLM): weighted input perplexity on LoRA-adapted model (see algorithms/TLM.md)."""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        labels = inputs["labels"]
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = out.logits

        nll = _per_sample_mean_nll(logits, labels)
        p_x = torch.exp(nll)
        lam = float(getattr(self.cfg, "lambda", 0.1))
        p0 = float(getattr(self.cfg, "p0", math.exp(3.0)))
        s_x = torch.where(p_x > p0, lam * (p_x / p0), torch.zeros_like(p_x))
        weighted = s_x * p_x
        batch_loss = weighted.sum()
        b = labels.size(0)

        if batch_loss > 0:
            loss = batch_loss / b
        else:
            loss = logits.float().sum() * 0.0

        return (loss, out) if return_outputs else loss
