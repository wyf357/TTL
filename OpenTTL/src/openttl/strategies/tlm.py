from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.base import Strategy
from openttl.strategies.tta_shared import apply_backbone_eval_lora_train, tta_model_forward


def _input_nll_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Input (self-supervised) NLL: mean over next-token CLM positions.

    ``labels`` are full sequence ids (prompt-only) from ``clm_full``; no gold labels.
    """
    B, T, V = logits.shape
    # CE = logsumexp(logits) - logit_of_label；恒等式写法避免 fp32 的 [B,T,V]
    # 物化（长序列下数 GB），bf16/fp16 均数值安全
    prev = logits[:, :-1, :]
    tgt = labels[:, 1:].clamp_min(0)  # -100 位置先用 0 占位，随后被 valid 掩掉
    lse = torch.logsumexp(prev, dim=-1)
    tok_logit = prev.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nll = lse - tok_logit
    valid = labels[:, 1:].ne(-100).float()
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (nll * valid).sum(dim=1) / denom


class TLMStrategy(Strategy):
    """Test-Time Learning (TLM): minimize S(x)·P(x) on unlabeled test input (see algorithms/TLM.md)."""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        apply_backbone_eval_lora_train(model)
        labels = inputs["labels"]
        if labels is not None:
            image_token_id = getattr(model.config, "image_token_id", None)
            if image_token_id is not None:
                labels = labels.clone()
                labels[labels == int(image_token_id)] = -100
        out = tta_model_forward(model, inputs)
        logits = out.logits

        nll = _input_nll_per_sample(logits, labels)
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
