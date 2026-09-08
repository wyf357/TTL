"""MMA 诊断：量化 δ 的量级、bf16 注入是否有效、以及对最终 logits 的实际影响。

用法: python diagnose_mma.py [sample_idx]
"""
import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from openttl.adapters.qwen3_5 import Qwen3_5Adapter
from openttl.strategies.mma import (
    find_language_layers, resolve_l_star, capture_layer_hidden,
)
from openttl.data.mmstar import load_mmstar_table

SYSTEM_PROMPT = (
    "You solve multiple-choice questions about images. "
    "Be precise and base your answer only on the provided image and question. "
    "Respond with exactly one chosen letter A, B, C, or D. "
    "After any internal reasoning, your final visible line must be exactly: Answer: X "
    "(X is A, B, C, or D)."
)

sample_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
device = torch.device("cuda")

cfg = SimpleNamespace(
    pretrained_model_name_or_path="/home/jxy/TTL/Qwen3.5-2B",
    revision=None, trust_remote_code=True,
    torch_dtype="bfloat16", attn_implementation="sdpa", device_map=None,
)
adapter = Qwen3_5Adapter()
adapter.load_processor(cfg)
model = adapter.load_model(cfg).to(device).eval()

ds = load_mmstar_table(local_root="/home/jxy/TTL/data/mmstar")
row = ds[sample_idx]
question = str(row["question"])
prompt = f"{question}\n\nAnswer with the letter only. Final Answer:"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
]
chat_text = adapter.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
enc = adapter.build_generate_inputs(
    prompt_text=chat_text, images=[row["image"]], device=device
)
input_ids = enc["input_ids"]
prompt_len = input_ids.shape[1]

image_token_id = int(getattr(model.config, "image_token_id", 248056))
vis_mask = input_ids[0] == image_token_id
vis_idx = vis_mask.nonzero(as_tuple=True)[0]

layers = find_language_layers(model)
L = len(layers)
ls = resolve_l_star(None, L)
layer_module = layers[ls - 1]

h = capture_layer_hidden(model, enc, layer_module)[0]  # [seq, d]
V, T = h[vis_mask], h[~vis_mask]
d = h.shape[1]
N_v, N_t = V.shape[0], T.shape[0]

print(f"L={L}  l*={ls}  d={d}  N_v={N_v}  N_t={N_t}  prompt_len={prompt_len}")
vn = V.float().norm(dim=1)
tn = T.float().norm(dim=1)
print(f"||V_j|| mean={vn.mean():.2f} median={vn.median():.2f} max={vn.max():.2f}")
print(f"||T_i|| mean={tn.mean():.2f} median={tn.median():.2f} max={tn.max():.2f}")

# ---- 带轨迹的 optimize_delta（与 mma.py 相同逻辑） ----
def optimize_tracked(V, T, K, eta, lam):
    V32, T32 = V.detach().float(), T.detach().float()
    T_norm = F.normalize(T32, p=2, dim=1)
    delta = torch.zeros(V32.shape[1], device=V32.device)
    hist = []
    for k in range(K):
        delta = delta.detach().requires_grad_(True)
        Vn = F.normalize(V32 + delta, p=2, dim=1)
        sim = (T_norm @ Vn.t()).mean()
        loss = -sim + lam * (delta ** 2).sum()
        g = torch.autograd.grad(loss, delta)[0]
        hist.append((k, sim.item(), loss.item(), g.norm().item()))
        with torch.no_grad():
            delta = delta - eta * g
    return delta.detach(), hist

t_bar = F.normalize(T.float(), p=2, dim=1).mean(0)  # 文本归一化中心
print(f"||t_bar|| (mean-cos 的理论上界) = {t_bar.norm().item():.4f}")

for eta in (0.1, 1.0, 10.0, 100.0):
    delta, hist = optimize_tracked(V, T, K=5, eta=eta, lam=0.01)
    sim0, simK = hist[0][1], hist[-1][1]
    dn = delta.norm().item()
    print(f"eta={eta:6.1f}: mean-cos {sim0:.6f} -> {simK:.6f} | "
          f"||grad|| step0={hist[0][3]:.3e} | ||delta||={dn:.3e} | "
          f"||delta||/mean||V||={dn / vn.mean().item():.2e}")

# ---- 生产配置下的 δ，bf16 注入有效性 ----
delta, hist = optimize_tracked(V, T, K=5, eta=0.1, lam=0.01)
d16 = delta.to(h.dtype).to(device)
V_new16 = (V + d16)
changed = (V_new16 != V).any(dim=1).float().mean().item()
elem_changed = (V_new16 != V).float().mean().item()
diff = (V_new16.float() - V.float()).abs()
print(f"\n[生产配置 K=5 eta=0.1 lam=0.01] ||delta||={delta.norm():.3e}")
print(f"bf16 注入后发生变化的视觉 token 比例: {changed:.4f}, 元素级变化比例: {elem_changed:.4f}")
print(f"bf16 实际注入量 max={diff.max():.3e} mean={diff.mean():.3e} (vs ||V|| mean={vn.mean():.1f})")

# ---- 对最终 logits 的影响 ----
def logits_with_inject(delta_inj):
    def _hook(module, args, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if out.shape[1] != prompt_len:
            return output
        hn = out.clone()
        hn[0, vis_idx] = hn[0, vis_idx] + delta_inj
        return (hn,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else hn
    handle = layer_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            return model(**enc).logits[0, -1].float()
    finally:
        handle.remove()

with torch.no_grad():
    base_logits = model(**enc).logits[0, -1].float()

for tag, dl in (("prod eta=0.1", delta), ("eta=100", optimize_tracked(V, T, 5, 100.0, 0.01)[0])):
    inj_logits = logits_with_inject(dl.to(h.dtype).to(device))
    dmax = (inj_logits - base_logits).abs().max().item()
    p, q = F.softmax(base_logits, -1), F.softmax(inj_logits, -1)
    kl = (q * (q.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum().item()
    same_top1 = int(inj_logits.argmax() == base_logits.argmax())
    print(f"[{tag}] logits max|diff|={dmax:.3e}  KL={kl:.3e}  top1不变={same_top1}")

# 零样本闭式变体的量级
v_bar = F.normalize(V.float(), p=2, dim=1).mean(0)
delta_zs = t_bar - v_bar
print(f"\n[zero-shot 闭式] ||delta_zs||={delta_zs.norm():.3e} "
      f"(相对 ||V||: {delta_zs.norm() / vn.mean():.2e})")
