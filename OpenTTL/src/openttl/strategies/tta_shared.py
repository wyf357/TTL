
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

try:
    from peft.tuners.lora.layer import LoraLayer
except ImportError:  # pragma: no cover
    LoraLayer = None

# 熵计算的序列分块大小（见 masked_mean_sequence_entropy）
_ENTROPY_CHUNK_TOKENS = 512


def tta_model_forward(model: nn.Module, inputs: Dict[str, Any]) -> Any:
    """单次 TTA 前向：关闭 KV cache，避免物化 ``past_key_values``（长序列显存占用可降数量级）。

    与 ``use_cache=True`` 相比，对整段 ``input_ids`` 一次前向得到的 ``logits`` 在数学上一致；
    仅影响是否分配/返回增量解码用的 cache 张量。
    """
    kwargs = {k: v for k, v in inputs.items() if k != "labels"}
    kwargs["use_cache"] = False
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        return model(**kwargs)


def apply_backbone_eval_lora_train(model: nn.Module) -> None:
    """PEFT：主干关闭 Dropout、仅 LoRA 子模块（含其 dropout）train。

    注意不能简单 ``model.eval()``：transformers 各层的梯度检查点条件是
    ``self.gradient_checkpointing and self.training``，``model.eval()`` 会把所有层
    的 ``training`` 置 False，梯度检查点静默失效 → 长序列（如文档整页 ~4000
    token）前向/反传保存全部激活而 OOM。Qwen 主干无 BN、Dropout p=0，
    保持 train 模式仅显式关掉 Dropout 模块，语义与 eval 等价但保留检查点。
    """
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    if LoraLayer is None:
        return
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.train()


def masked_mean_sequence_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    eps: float,  # noqa: ARG001 — 保留参数兼容；新公式无需 eps
) -> torch.Tensor:
    """序列内掩码均值熵，再对 batch 均值（tent.md 公式；EATA 复用聚合方式）。

    数值与显存安全的热力学恒等式写法：
        ent = -(p * logp).sum，其中 logp = logits - logsumexp(logits)
    - 不再有 ``log(p+eps)``：fp16 下 eps 舍入为 0 导致 0×(-inf)=NaN（旧 bug）；
    - 不升 fp32：长序列下 fp32 的 [B,T,V] logits/softmax/logp 会占数 GB 显存，
      p=0 时 p*logp = 0×有限值 = 0，bf16/fp16 均安全；
    - 沿序列维分块 + 逐块 checkpoint：全序列的 softmax/logp 及其反传中间量
      （约 6×[B,T,V]）是长序列 OOM 的主因，分块后峰值只剩一块的大小，
      数学上与整块计算完全等价。
    """

    def _chunk_ent_sum(lc: torch.Tensor, mc: torch.Tensor) -> torch.Tensor:
        lse = torch.logsumexp(lc, dim=-1, keepdim=True)
        p = F.softmax(lc, dim=-1)
        logp = lc - lse
        ent = -(p * logp).sum(dim=-1)
        return (ent * mc).sum(dim=1)  # [B]

    m = mask.bool().to(dtype=logits.dtype, device=logits.device)
    t_len = logits.shape[1]
    chunk = int(_ENTROPY_CHUNK_TOKENS)
    nums: list[torch.Tensor] = []
    for s in range(0, t_len, chunk):
        lc = logits[:, s : s + chunk]
        mc = m[:, s : s + chunk]
        if logits.requires_grad:
            nums.append(
                torch.utils.checkpoint.checkpoint(_chunk_ent_sum, lc, mc, use_reentrant=False)
            )
        else:
            nums.append(_chunk_ent_sum(lc, mc))
    num = torch.stack(nums, dim=0).sum(dim=0)  # [B]
    den = m.sum(dim=1).clamp_min(1.0)
    per_seq = num / den
    if per_seq.numel() == 0:
        return logits.float().sum() * 0.0
    return per_seq.mean()
