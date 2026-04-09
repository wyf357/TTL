
from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.base import Strategy
from openttl.strategies.tta_shared import apply_backbone_eval_lora_train


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """在 vocab 维上裁剪 logits，供 Softmax 后 multinomial 采样（come.md Best Practices）。"""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = probs.cumsum(dim=-1)
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """对原始 logits 做温度 / top-k / top-p 后多项式采样，形状 [B]。"""
    if temperature > 0:
        logits = logits / temperature
    logits = _apply_top_k_top_p(logits, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any():
        probs = F.softmax(torch.zeros_like(logits), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _pad_prompt_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 attention_mask 取每条样本非 pad 段，右 pad 到 batch 内最大 prompt 长度。"""
    device = input_ids.device
    dtype_ids = input_ids.dtype
    dtype_am = attention_mask.dtype
    b = input_ids.size(0)
    rows = []
    lengths = []
    for i in range(b):
        sl = int(attention_mask[i].sum().item())
        sl = max(sl, 1)
        rows.append(input_ids[i, :sl])
        lengths.append(sl)
    max_len = max(lengths)
    ids = torch.full((b, max_len), pad_token_id, device=device, dtype=dtype_ids)
    am = torch.zeros((b, max_len), device=device, dtype=dtype_am)
    for i, row in enumerate(rows):
        L = row.size(0)
        ids[i, :L] = row
        am[i, :L] = 1
    return ids, am


def _opinion_entropy(
    logits_come: torch.Tensor,
    vocab_size: int,
    eps: float,
) -> torch.Tensor:
    """H(M) 按 come.md：belief 与 uncertainty 两项，返回形状 [B]。"""
    evidence = torch.exp(logits_come)
    s = evidence.sum(dim=-1, keepdim=True) + float(vocab_size)
    belief = evidence / s
    u = float(vocab_size) / s.squeeze(-1)
    h_belief = -(belief * torch.log(belief + eps)).sum(dim=-1)
    h_u = -u * torch.log(u + eps)
    return h_belief + h_u


class COMEStrategy(Strategy):
    """COME-LLM：保守最小化意见熵 + logit 范数约束；自回归 rollout（algorithms/come.md）。"""

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        apply_backbone_eval_lora_train(model)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pad_token_id = int(
            getattr(model.config, "pad_token_id", None) or 0,
        )

        t_steps = max(1, int(getattr(self.cfg, "rollout_steps", 16)))
        p_norm = float(getattr(self.cfg, "norm_p", 2.0))
        tau = float(getattr(self.cfg, "tau", 1.0))
        eps = float(getattr(self.cfg, "epsilon", 1e-8))
        temperature = float(getattr(self.cfg, "temperature", 1.0))
        top_k = int(getattr(self.cfg, "top_k", 0))
        top_p = float(getattr(self.cfg, "top_p", 1.0))

        vocab_size = int(model.config.vocab_size)
        ids, am = _pad_prompt_batch(input_ids, attention_mask, pad_token_id)

        total_h: torch.Tensor | None = None
        last_out = None
        ord_: float | int = float("inf") if math.isinf(p_norm) else p_norm

        for _ in range(t_steps):
            fwd = {"input_ids": ids, "attention_mask": am}
            last_out = model(**fwd)
            logits_z = last_out.logits[:, -1, :]

            norm_z = torch.linalg.vector_norm(logits_z, ord=ord_, dim=-1, keepdim=True)
            logits_come = tau * norm_z.detach() * logits_z / (norm_z + eps)

            h = _opinion_entropy(logits_come, vocab_size, eps)
            step_mean = h.mean()
            total_h = step_mean if total_h is None else total_h + step_mean

            next_tok = _sample_next_token(logits_z, temperature, top_k, top_p)
            ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
            am = torch.cat(
                [am, torch.ones(ids.size(0), 1, device=am.device, dtype=am.dtype)],
                dim=1,
            )

        assert total_h is not None
        loss = total_h / float(t_steps)
        return (loss, last_out) if return_outputs else loss
