
from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.base import Strategy
from openttl.strategies.tta_shared import apply_backbone_eval_lora_train, tta_model_forward


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
    # fp16 下 exp(logits)（|z| 可达 30+）会溢出，升精度计算
    evidence = torch.exp(logits_come.float())
    s = evidence.sum(dim=-1, keepdim=True) + float(vocab_size)
    belief = evidence / s
    u = float(vocab_size) / s.squeeze(-1)
    h_belief = -(belief * torch.log(belief + eps)).sum(dim=-1)
    h_u = -u * torch.log(u + eps)
    return h_belief + h_u


class COMEStrategy(Strategy):
    """COME-LLM：保守最小化意见熵 + logit 范数约束；自回归 rollout（algorithms/come.md）。

    显存说明：T 步 rollout 的熵均值等价于各步熵/T 的梯度累积，因此本策略在
    ``compute_loss`` 内逐步 backward（峰值显存 = 单步图），并返回 detached 标量；
    调用方（``OnlineTTARunner``）通过 ``handles_own_backward`` 跳过外层 backward。
    """

    handles_own_backward = True

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        apply_backbone_eval_lora_train(model)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # 多模态顶层 config（如 Qwen3_5Config）没有 vocab_size / pad_token_id，
        # 需回退到 text_config。
        _mcfg = model.config
        _text_cfg = getattr(_mcfg, "text_config", None)
        pad_token_id = int(
            getattr(_mcfg, "pad_token_id", None)
            or getattr(_text_cfg, "pad_token_id", None)
            or 0
        )

        t_steps = max(1, int(getattr(self.cfg, "rollout_steps", 16)))
        p_norm = float(getattr(self.cfg, "norm_p", 2.0))
        tau = float(getattr(self.cfg, "tau", 1.0))
        eps = float(getattr(self.cfg, "epsilon", 1e-8))
        temperature = float(getattr(self.cfg, "temperature", 1.0))
        top_k = int(getattr(self.cfg, "top_k", 0))
        top_p = float(getattr(self.cfg, "top_p", 1.0))

        vocab_size = int(
            getattr(_mcfg, "vocab_size", None)
            or getattr(_text_cfg, "vocab_size")
        )
        ids, am = _pad_prompt_batch(input_ids, attention_mask, pad_token_id)

        # Multimodal tensors (e.g. pixel_values): needed on EVERY rollout step —
        # use_cache=False，每步都对整段序列重新前向，图像 token 位置必须由视觉特征填充。
        # mm_token_type_ids 随新采样 token 追加 text 类型(0)。
        extra_mm: Dict[str, Any] = {}
        for k, v in inputs.items():
            if k in ("input_ids", "attention_mask", "labels"):
                continue
            if isinstance(v, torch.Tensor):
                extra_mm[k] = v

        total_h_val = 0.0
        last_out = None
        ord_: float | int = float("inf") if math.isinf(p_norm) else p_norm

        for step in range(t_steps):
            # 只需末位 logits；logits_to_keep=1 避免物化 [B, T, V] 全量 logits
            fwd: Dict[str, Any] = {
                "input_ids": ids,
                "attention_mask": am,
                "logits_to_keep": 1,
            }
            fwd.update(extra_mm)
            last_out = tta_model_forward(model, fwd)
            logits_z = last_out.logits[:, -1, :]

            norm_z = torch.linalg.vector_norm(logits_z, ord=ord_, dim=-1, keepdim=True)
            logits_come = tau * norm_z.detach() * logits_z / (norm_z + eps)

            h = _opinion_entropy(logits_come, vocab_size, eps)
            # 逐步 backward（梯度累积）：与对 T 步熵均值一次 backward 数学等价，
            # 但峰值显存只有单步计算图。
            step_loss = h.mean() / float(t_steps)
            step_loss.backward()
            total_h_val += float(step_loss.detach())

            next_tok = _sample_next_token(logits_z.detach(), temperature, top_k, top_p)
            ids = torch.cat([ids, next_tok.unsqueeze(1)], dim=1)
            am = torch.cat(
                [am, torch.ones(ids.size(0), 1, device=am.device, dtype=am.dtype)],
                dim=1,
            )
            mtt = extra_mm.get("mm_token_type_ids")
            if mtt is not None:
                extra_mm["mm_token_type_ids"] = torch.cat(
                    [
                        mtt,
                        torch.zeros(mtt.size(0), 1, device=mtt.device, dtype=mtt.dtype),
                    ],
                    dim=1,
                )

        loss = torch.tensor(total_h_val, device=input_ids.device)
        return (loss, last_out) if return_outputs else loss
