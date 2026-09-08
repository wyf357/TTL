"""MMA: Modality Mirror Alignment（模态镜像对齐）测试时对齐。

与 TENT/TLM 等 LoRA-TTA 策略不同，MMA 完全冻结模型参数，唯一优化变量是
一个 d 维模态偏移向量 δ：

1. 前向至目标层 l*（推荐 floor(2L/3)），提取视觉/文本 token 的冻结隐状态
   V^{(l*)}、T^{(l*)}；
2. 以「最大化归一化视觉 token（加偏移后）与归一化文本 token 的平均余弦
   相似度 + L2 正则」为目标，对 δ 做 K 步梯度下降（autograd，全程不反传
   进模型）；
3. 在 l* 层挂上 forward hook，把优化后的 δ 加到视觉 token 的隐状态上，
   再执行常规 generate（prefill 等价于从 l*+1 层继续前向）。

实现要点：
- Qwen3.5 的 ``Qwen3_5TextModel.forward`` 不支持 ``output_hidden_states``，
  因此用 forward hook 捕获 l* 层输出（decoder layer 直接返回 tensor）。
- hook 只在 prefill（seq_len == prompt_len）时施加偏移；decode 步 seq_len=1，
  自然跳过。
- δ 优化在 float32 中进行，注入时 cast 回模型 dtype。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def find_language_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """定位语言模型的 decoder layer 列表（Qwen3.5: model.model.language_model.layers）。"""
    # 常见路径优先
    for path in (
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.layers",
        "model.layers",
    ):
        mod: Any = model
        ok = True
        for part in path.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                ok = False
                break
        if ok and isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            return mod
    # 兜底：找最长的同名 ModuleList
    best: Optional[torch.nn.ModuleList] = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("无法在模型中定位 decoder layers (ModuleList)")
    return best


def resolve_l_star(l_star: Optional[int], num_layers: int) -> int:
    """l* 为 1-based 层索引；None/<=0 时取 floor(2L/3)。"""
    if l_star is None or int(l_star) <= 0:
        return max(1, math.floor(2 * num_layers / 3))
    return min(int(l_star), num_layers)


@torch.no_grad()
def capture_layer_hidden(
    model: torch.nn.Module,
    enc: Dict[str, Any],
    layer_module: torch.nn.Module,
) -> torch.Tensor:
    """前向一次并用 hook 捕获 l* 层输出的完整隐状态 [1, seq, d]（detached）。"""
    store: Dict[str, torch.Tensor] = {}

    def _hook(module: torch.nn.Module, args: Any, output: Any) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        store["h"] = out.detach()

    handle = layer_module.register_forward_hook(_hook)
    try:
        model(**enc, use_cache=False)
    finally:
        handle.remove()
    if "h" not in store:
        raise RuntimeError("hook 未捕获到隐状态，请检查 layer_module 是否正确")
    return store["h"]


def optimize_delta(
    V: torch.Tensor,
    T: torch.Tensor,
    *,
    K: int = 5,
    eta: float = 0.1,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """对 δ 做 K 步梯度下降；输入隐状态先转 float32，返回 float32 δ [d]。

    L(δ) = -mean_{i,j} cos(T_i, V_j + δ) + λ||δ||^2
    """
    V32 = V.detach().to(torch.float32)  # [N_v, d]
    T32 = T.detach().to(torch.float32)  # [N_t, d]
    T_norm = F.normalize(T32, p=2, dim=1)

    delta = torch.zeros(V32.shape[1], device=V32.device, dtype=torch.float32)
    for _ in range(int(K)):
        delta = delta.detach().requires_grad_(True)
        V_prime = V32 + delta  # 广播: [N_v, d] + [d]
        V_norm = F.normalize(V_prime, p=2, dim=1)
        mean_sim = (T_norm @ V_norm.t()).mean()
        loss = -mean_sim + float(lambda_reg) * (delta**2).sum()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - float(eta) * grad
    return delta.detach()


def mma_generate(
    *,
    model: torch.nn.Module,
    adapter: Any,
    prompt_text: str,
    images: Optional[Sequence[Any]],
    device: torch.device,
    l_star: Optional[int] = None,
    K: int = 5,
    eta: float = 0.1,
    lambda_reg: float = 0.01,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    image_token_id: Optional[int] = None,
) -> str:
    """单样本 MMA：优化 δ 后在 l* 层注入偏移并 generate，返回解码文本。"""
    enc = adapter.build_generate_inputs(
        prompt_text=prompt_text, images=list(images) if images else None, device=device
    )
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[1])

    if image_token_id is None:
        image_token_id = int(getattr(model.config, "image_token_id", 248056))
    vis_mask = input_ids[0] == int(image_token_id)  # [seq] bool
    n_vis = int(vis_mask.sum().item())
    if n_vis == 0:
        raise RuntimeError("MMA 需要视觉 token，但 input_ids 中未找到 image_token_id")

    layers = find_language_layers(model)
    ls = resolve_l_star(l_star, len(layers))
    layer_module = layers[ls - 1]  # l* 为 1-based；hidden_states[l*] == layers[l*-1] 输出

    model.eval()
    # Step 1: 前向至 l* 层并冻结提取 V、T
    h = capture_layer_hidden(model, enc, layer_module)  # [1, seq, d]
    h0 = h[0]
    V = h0[vis_mask]
    T = h0[~vis_mask]

    # Step 2: 测试时优化 δ（不触碰模型参数）
    delta32 = optimize_delta(V, T, K=K, eta=eta, lambda_reg=lambda_reg)
    delta_inj = delta32.to(h0.dtype).to(device)
    vis_idx = vis_mask.nonzero(as_tuple=True)[0]

    # Step 3: 挂 hook，在 prefill 时把 δ 加到视觉 token 的 l* 层输出上
    def _inject_hook(module: torch.nn.Module, args: Any, output: Any) -> Any:
        if isinstance(output, (tuple, list)):
            h_in = output[0]
            if h_in.shape[1] != prompt_len:
                return output
            h_new = h_in.clone()
            h_new[0, vis_idx] = h_new[0, vis_idx] + delta_inj
            return (h_new,) + tuple(output[1:])
        if output.shape[1] != prompt_len:
            return output
        h_new = output.clone()
        h_new[0, vis_idx] = h_new[0, vis_idx] + delta_inj
        return h_new

    handle = layer_module.register_forward_hook(_inject_hook)
    tok = adapter.tokenizer()
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
    do_sample = temperature > 0
    try:
        with torch.no_grad():
            gkw: Dict[str, Any] = dict(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                pad_token_id=pad_id,
            )
            if do_sample:
                gkw["temperature"] = temperature
                gkw["top_p"] = top_p
            gen_ids = model.generate(**gkw)
    finally:
        handle.remove()
    return adapter.decode_new_tokens(gen_ids, prompt_len).strip()
