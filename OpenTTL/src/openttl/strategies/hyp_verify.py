"""假设验证共用工具：问题条件化视觉门控（H0/H1/H2 实验）。

不接入 run_mmstar；供 verify_hyp_*.py 脚本调用。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from openttl.strategies.mma import (
    capture_layer_hidden,
    find_language_layers,
    optimize_delta,
    resolve_l_star,
)

SYSTEM_PROMPT = (
    "You solve multiple-choice questions about images. "
    "Be precise and base your answer only on the provided image and question. "
    "Respond with exactly one chosen letter A, B, C, or D. "
    "After any internal reasoning, your final visible line must be exactly: Answer: X "
    "(X is A, B, C, or D)."
)

DEFAULT_MODEL_PATH = "/home/jxy/TTL/Qwen3.5-2B"
DEFAULT_MMSTAR_ROOT = "/home/jxy/TTL/data/mmstar"
DEFAULT_BASELINE_JSON = "/home/jxy/TTL/OpenTTL/outputs/mmstar_mma/baseline_metrics.json"
HYP_OUTPUT_DIR = Path("/home/jxy/TTL/OpenTTL/outputs/mmstar_hyp")


def default_model_cfg(model_path: str = DEFAULT_MODEL_PATH) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=model_path,
        revision=None,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        device_map=None,
    )


def format_mmstar_prompt(question: str) -> str:
    return f"{question}\n\nAnswer with the letter only. Final Answer:"


def build_mmstar_messages(question: str, *, embed_image: bool = False) -> List[Dict[str, Any]]:
    img_content: Any = {"type": "image", "image": None} if embed_image else {"type": "image"}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [img_content, {"type": "text", "text": format_mmstar_prompt(question)}],
        },
    ]


def image_token_id(model: torch.nn.Module) -> int:
    return int(getattr(model.config, "image_token_id", 248056))


def visual_mask(input_ids: torch.Tensor, img_tok_id: int) -> torch.Tensor:
    return input_ids[0] == int(img_tok_id)


def question_mask(
    input_ids: torch.Tensor,
    vis_mask: torch.Tensor,
    special_ids: Sequence[int],
) -> torch.Tensor:
    """题干 Q：最后一个视觉 token 之后、排除 special 的用户文本 token。"""
    seq_len = int(input_ids.shape[1])
    vis_pos = vis_mask.nonzero(as_tuple=True)[0]
    if vis_pos.numel() == 0:
        return torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    last_vis = int(vis_pos.max().item())
    special = set(int(x) for x in special_ids)
    ids = input_ids[0].tolist()
    q_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    for pos in range(last_vis + 1, seq_len):
        if ids[pos] not in special:
            q_mask[pos] = True
    return q_mask


def nonvisual_text_mask(input_ids: torch.Tensor, vis_mask: torch.Tensor) -> torch.Tensor:
    """MMA 口径：全部非视觉 token。"""
    return ~vis_mask


@dataclass
class SampleContext:
    enc: Dict[str, Any]
    prompt_len: int
    vis_mask: torch.Tensor
    q_mask: torch.Tensor
    all_txt_mask: torch.Tensor
    vis_idx: torch.Tensor
    layers: torch.nn.ModuleList
    l_star: int
    score_layer: torch.nn.Module


def prepare_sample_context(
    model: torch.nn.Module,
    adapter: Any,
    *,
    question: str,
    image: Any,
    device: torch.device,
    l_star: Optional[int] = None,
) -> SampleContext:
    messages = build_mmstar_messages(question)
    chat_text = adapter.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = adapter.build_generate_inputs(
        prompt_text=chat_text, images=[image], device=device
    )
    prompt_len = int(enc["input_ids"].shape[1])
    img_id = image_token_id(model)
    vmask = visual_mask(enc["input_ids"], img_id)
    tok = adapter.tokenizer()
    special_ids = list(getattr(tok, "all_special_ids", []) or [])
    qmask = question_mask(enc["input_ids"], vmask, special_ids)
    layers = find_language_layers(model)
    ls = resolve_l_star(l_star, len(layers))
    return SampleContext(
        enc=enc,
        prompt_len=prompt_len,
        vis_mask=vmask,
        q_mask=qmask,
        all_txt_mask=nonvisual_text_mask(enc["input_ids"], vmask),
        vis_idx=vmask.nonzero(as_tuple=True)[0],
        layers=layers,
        l_star=ls,
        score_layer=layers[ls - 1],
    )


def compute_relevance(
    h: torch.Tensor,
    vis_mask: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    mode: str = "max",
) -> torch.Tensor:
    """r_j：文本 token 与视觉 token 的余弦相关度（per visual token）。"""
    V = h[vis_mask].float()
    T = h[text_mask].float()
    if V.numel() == 0 or T.numel() == 0:
        return torch.zeros(int(vis_mask.sum()), device=h.device)
    Vn = F.normalize(V, p=2, dim=1)
    Tn = F.normalize(T, p=2, dim=1)
    sim = Tn @ Vn.t()
    if mode == "mean":
        return sim.mean(dim=0)
    return sim.max(dim=0).values


def zscore_gates(
    scores: torch.Tensor,
    *,
    alpha: float = 0.5,
    clip_c: float = 2.0,
    invert: bool = False,
) -> torch.Tensor:
    s = scores.float()
    z = (s - s.mean()) / (s.std() + 1e-6)
    z = z.clamp(-clip_c, clip_c)
    sign = -1.0 if invert else 1.0
    return 1.0 + sign * float(alpha) * z


def shuffle_gates(gates: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    perm = torch.randperm(gates.numel(), generator=generator)
    return gates[perm.to(gates.device)]


def uniform_gates(n_vis: int, *, alpha: float, device: torch.device) -> torch.Tensor:
    return torch.ones(n_vis, device=device) * (1.0 + float(alpha))


def random_zscore_gates(
    n_vis: int,
    *,
    alpha: float = 0.5,
    clip_c: float = 2.0,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    z = torch.randn(n_vis, generator=generator)
    z = z.to(device)
    z = z.clamp(-clip_c, clip_c)
    return 1.0 + float(alpha) * z


def spikiness_metrics(r: torch.Tensor, tau: Optional[float] = None) -> Dict[str, float]:
    s = r.float()
    if s.numel() == 0:
        return dict(entropy=0.0, peak_gap=0.0, top10_mass=0.0, cv=0.0)
    if tau is None:
        tau = max(float(s.std()), 1e-6)
    p = F.softmax(s / tau, dim=0)
    ent = float(-(p * p.clamp_min(1e-12).log()).sum().item())
    med = float(s.median().item())
    peak_gap = float(s.max().item() - med)
    k10 = min(10, s.numel())
    top10_mass = float(torch.topk(p, k10).values.sum().item())
    cv = float(s.std().item() / (s.mean().item() + 1e-12))
    return dict(entropy=ent, peak_gap=peak_gap, top10_mass=top10_mass, cv=cv)


def topk_iou(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, a.numel(), b.numel())
    if k <= 0:
        return 0.0
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    inter = len(ia & ib)
    union = len(ia | ib)
    return inter / union if union else 0.0


def random_iou_baseline(n: int, k: int, trials: int = 20) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    vals = []
    for _ in range(trials):
        a = torch.randperm(n)[:k]
        b = torch.randperm(n)[:k]
        ia, ib = set(a.tolist()), set(b.tolist())
        union = len(ia | ib)
        vals.append(len(ia & ib) / union if union else 0.0)
    return float(sum(vals) / len(vals))


@torch.no_grad()
def forward_last_logits(model: torch.nn.Module, enc: Dict[str, Any]) -> torch.Tensor:
    return model(**enc, use_cache=False).logits[0, -1].float()


def logits_delta_stats(
    base_logits: torch.Tensor,
    inj_logits: torch.Tensor,
) -> Dict[str, float]:
    dmax = float((inj_logits - base_logits).abs().max().item())
    p = F.softmax(base_logits, dim=-1)
    q = F.softmax(inj_logits, dim=-1)
    kl = float((q * (q.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum().item())
    same_top1 = int(inj_logits.argmax() == base_logits.argmax())
    return dict(max_abs_diff=dmax, kl=kl, same_top1=same_top1)


class GateHookManager:
    """在 l_star..L 各层 input_layernorm 上注入逐视觉 token 门控。"""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        *,
        l_star_1based: int,
        prompt_len: int,
        vis_idx: torch.Tensor,
        gates: torch.Tensor,
        pre_ln: bool = False,
        n_inject_layers: Optional[int] = None,
    ) -> None:
        self.layers = layers
        self.l_start = l_star_1based - 1
        self.prompt_len = prompt_len
        self.vis_idx = vis_idx
        self.gates = gates.detach()
        self.pre_ln = pre_ln
        self.n_inject = n_inject_layers or (len(layers) - self.l_start)
        self._handles: List[Any] = []

    def _apply(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] != self.prompt_len:
            return hidden
        h_new = hidden.clone()
        g = self.gates.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        h_new[0, self.vis_idx] = h_new[0, self.vis_idx] * g
        return h_new

    def _post_hook(self, module: torch.nn.Module, args: Any, output: Any) -> Any:
        if isinstance(output, (tuple, list)):
            h_new = self._apply(output[0])
            return (h_new,) + tuple(output[1:])
        return self._apply(output)

    def _pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not args:
            return args
        hidden = args[0]
        h_new = self._apply(hidden)
        return (h_new,) + tuple(args[1:])

    def register(self) -> None:
        end = min(self.l_start + self.n_inject, len(self.layers))
        for layer_idx in range(self.l_start, end):
            rms = self.layers[layer_idx].input_layernorm
            if self.pre_ln:
                self._handles.append(rms.register_forward_pre_hook(self._pre_hook))
            else:
                self._handles.append(rms.register_forward_hook(self._post_hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class DeltaHookManager:
    """MMA 式：在 l* 层输出上对视觉 token 加同一 δ。"""

    def __init__(
        self,
        layer_module: torch.nn.Module,
        *,
        prompt_len: int,
        vis_idx: torch.Tensor,
        delta: torch.Tensor,
    ) -> None:
        self.layer_module = layer_module
        self.prompt_len = prompt_len
        self.vis_idx = vis_idx
        self.delta = delta.detach()
        self._handle: Any = None

    def _hook(self, module: torch.nn.Module, args: Any, output: Any) -> Any:
        if isinstance(output, (tuple, list)):
            h_in = output[0]
            if h_in.shape[1] != self.prompt_len:
                return output
            h_new = h_in.clone()
            d = self.delta.to(device=h_in.device, dtype=h_in.dtype)
            h_new[0, self.vis_idx] = h_new[0, self.vis_idx] + d
            return (h_new,) + tuple(output[1:])
        if output.shape[1] != self.prompt_len:
            return output
        h_new = output.clone()
        d = self.delta.to(device=output.device, dtype=output.dtype)
        h_new[0, self.vis_idx] = h_new[0, self.vis_idx] + d
        return h_new

    def register(self) -> None:
        self._handle = self.layer_module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def measure_intervention_logits(
    model: torch.nn.Module,
    enc: Dict[str, Any],
    base_logits: torch.Tensor,
    manager: Any,
) -> Dict[str, float]:
    manager.register()
    try:
        inj_logits = forward_last_logits(model, enc)
    finally:
        manager.remove()
    return logits_delta_stats(base_logits, inj_logits)


def mma_production_delta(
    h: torch.Tensor,
    vis_mask: torch.Tensor,
    *,
    K: int = 5,
    eta: float = 0.1,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    V = h[vis_mask]
    T = h[~vis_mask]
    return optimize_delta(V, T, K=K, eta=eta, lambda_reg=lambda_reg)


def compute_gold_saliency(
    model: torch.nn.Module,
    enc: Dict[str, Any],
    layer_module: torch.nn.Module,
    vis_mask: torch.Tensor,
    gold_letter: str,
    tokenizer: Any,
) -> Optional[torch.Tensor]:
    """|∂ logit_gold / ∂ h_j|，仅视觉 token（l* 层输出）。"""
    stored: Dict[str, torch.Tensor] = {}

    def _hook(module: torch.nn.Module, args: Any, output: Any) -> Any:
        out = output[0] if isinstance(output, (tuple, list)) else output
        out.retain_grad()
        stored["h"] = out
        return output

    handle = layer_module.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            outputs = model(**enc, use_cache=False)
    except torch.cuda.OutOfMemoryError:
        handle.remove()
        torch.cuda.empty_cache()
        return None
    finally:
        handle.remove()

    if "h" not in stored:
        return None
    h = stored["h"]
    logits = outputs.logits[0, -1]
    gold_ids = tokenizer.encode(str(gold_letter).strip().upper(), add_special_tokens=False)
    if not gold_ids:
        del outputs, h, stored
        return None
    gold_id = int(gold_ids[-1])
    model.zero_grad(set_to_none=True)
    try:
        logits[gold_id].backward(retain_graph=False)
    except torch.cuda.OutOfMemoryError:
        del outputs, h, stored
        torch.cuda.empty_cache()
        return None
    if h.grad is None:
        del outputs, h, stored
        return None
    sal = h.grad[0, vis_mask].detach().float().abs().sum(dim=-1).cpu()
    h.grad = None
    model.zero_grad(set_to_none=True)
    del outputs, h, stored, logits
    return sal


def gates_for_condition(
    condition: str,
    r: torch.Tensor,
    *,
    alpha: float,
    clip_c: float = 2.0,
    generator: Optional[torch.Generator] = None,
    saliency: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cond = condition.lower()
    if cond == "q_aligned":
        return zscore_gates(r, alpha=alpha, clip_c=clip_c)
    if cond == "q_anti":
        return zscore_gates(r, alpha=alpha, clip_c=clip_c, invert=True)
    if cond == "shuffle":
        g = zscore_gates(r, alpha=alpha, clip_c=clip_c)
        return shuffle_gates(g, generator=generator)
    if cond == "uniform":
        return uniform_gates(r.numel(), alpha=alpha, device=r.device)
    if cond == "random":
        return random_zscore_gates(
            r.numel(), alpha=alpha, clip_c=clip_c, device=r.device, generator=generator
        )
    if cond == "oracle":
        if saliency is None:
            raise ValueError("oracle 条件需要 saliency")
        return zscore_gates(saliency, alpha=alpha, clip_c=clip_c)
    raise ValueError(f"未知门控条件: {condition}")


def generate_with_gates(
    model: torch.nn.Module,
    adapter: Any,
    ctx: SampleContext,
    gates: torch.Tensor,
    *,
    pre_ln: bool = False,
    max_new_tokens: int = 1024,
) -> str:
    mgr = GateHookManager(
        ctx.layers,
        l_star_1based=ctx.l_star,
        prompt_len=ctx.prompt_len,
        vis_idx=ctx.vis_idx,
        gates=gates,
        pre_ln=pre_ln,
    )
    mgr.register()
    tok = adapter.tokenizer()
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
    try:
        with torch.no_grad():
            gen_ids = model.generate(
                **ctx.enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=pad_id,
            )
    finally:
        mgr.remove()
    return adapter.decode_new_tokens(gen_ids, ctx.prompt_len).strip()


def load_selected_alpha(path: Optional[Path] = None) -> float:
    p = path or (HYP_OUTPUT_DIR / "selected_alpha.json")
    if p.is_file():
        data = json.loads(p.read_text())
        return float(data.get("alpha", 0.5))
    return 0.5


def save_selected_alpha(alpha: float, path: Optional[Path] = None) -> Path:
    HYP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = path or (HYP_OUTPUT_DIR / "selected_alpha.json")
    p.write_text(json.dumps({"alpha": alpha}, indent=2))
    return p


def paired_mcnemar(y_a: Sequence[bool], y_b: Sequence[bool]) -> Dict[str, float]:
    """配对 McNemar（两条件对错比较）。"""
    import numpy as np
    from scipy import stats

    a = np.asarray(y_a, dtype=bool)
    b = np.asarray(y_b, dtype=bool)
    assert len(a) == len(b)
    # b 对 a 错 -> 01; a 对 b 错 -> 10
    b01 = int(np.sum((~a) & b))
    b10 = int(np.sum(a & (~b)))
    if b01 + b10 == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue)
    return dict(
        n=len(a),
        acc_a=float(a.mean()),
        acc_b=float(b.mean()),
        delta_acc=float(b.mean() - a.mean()),
        b01=b01,
        b10=b10,
        p_value=p,
    )
