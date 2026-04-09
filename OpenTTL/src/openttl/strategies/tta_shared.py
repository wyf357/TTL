
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from peft.tuners.lora.layer import LoraLayer
except ImportError:  # pragma: no cover
    LoraLayer = None


def apply_backbone_eval_lora_train(model: nn.Module) -> None:
    """PEFT：主干 eval，仅 LoRA 子模块 train（与 tent.md / EATA 共用）。"""
    model.eval()
    if LoraLayer is None:
        return
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.train()


def masked_mean_sequence_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """序列内掩码均值熵，再对 batch 均值（tent.md 公式；EATA 复用聚合方式）。"""
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(probs + eps)
    token_ent = -(probs * logp).sum(dim=-1)
    m = mask.bool().to(dtype=logits.dtype, device=logits.device)
    num = (token_ent * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1.0)
    per_seq = num / den
    if per_seq.numel() == 0:
        return logits.float().sum() * 0.0
    return per_seq.mean()
