"""相关性测试：baseline mean-cos（MMA 目标函数值）与逐题对错是否相关。

复用 outputs/mmstar_mma/baseline_metrics.json 的逐题结果，仅对每个样本做
一次前向、取 l*=16 层隐状态计算 mean-cos，不做任何生成。
"""
import os, sys, json, csv
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from openttl.adapters.qwen3_5 import Qwen3_5Adapter
from openttl.strategies.mma import find_language_layers, resolve_l_star, capture_layer_hidden
from openttl.data.mmstar import load_mmstar_table

SYSTEM_PROMPT = (
    "You solve multiple-choice questions about images. "
    "Be precise and base your answer only on the provided image and question. "
    "Respond with exactly one chosen letter A, B, C, or D. "
    "After any internal reasoning, your final visible line must be exactly: Answer: X "
    "(X is A, B, C, or D)."
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT_CSV = "/home/jxy/TTL/OpenTTL/outputs/mmstar_mma/mean_cos_corr.csv"
BASELINE = "/home/jxy/TTL/OpenTTL/outputs/mmstar_mma/baseline_metrics.json"
device = torch.device("cuda")

results = json.load(open(BASELINE))["results"]

cfg = SimpleNamespace(
    pretrained_model_name_or_path="/home/jxy/TTL/Qwen3.5-2B",
    revision=None, trust_remote_code=True,
    torch_dtype="bfloat16", attn_implementation="sdpa", device_map=None,
)
adapter = Qwen3_5Adapter()
adapter.load_processor(cfg)
model = adapter.load_model(cfg).to(device).eval()

ds = load_mmstar_table(local_root="/home/jxy/TTL/data/mmstar")
layers = find_language_layers(model)
ls = resolve_l_star(None, len(layers))
layer_module = layers[ls - 1]
image_token_id = int(getattr(model.config, "image_token_id", 248056))
print(f"l*={ls}, N={N}", flush=True)

rows = []
for i in range(N):
    row = ds[i]
    res = results[i]
    assert res["idx"] == i, f"index mismatch at {i}: {res['idx']}"
    q = str(row["question"])
    assert q[:100] == res["question"][:100] or res["question"].startswith(q[:80]), f"question mismatch at {i}"

    prompt = f"{q}\n\nAnswer with the letter only. Final Answer:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    chat_text = adapter.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = adapter.build_generate_inputs(prompt_text=chat_text, images=[row["image"]], device=device)
    vis_mask = enc["input_ids"][0] == image_token_id
    h = capture_layer_hidden(model, enc, layer_module)[0].float()
    V, T = h[vis_mask], h[~vis_mask]

    Vn, Tn = F.normalize(V, p=2, dim=1), F.normalize(T, p=2, dim=1)
    mean_cos = (Tn @ Vn.t()).mean().item()                      # MMA 目标函数值
    cos_vv = (Vn @ Vn.t()).mean().item()                        # 视觉内部自相似（含对角）
    cos_tt = (Tn @ Tn.t()).mean().item()                        # 文本内部自相似
    rows.append(dict(
        idx=i, mean_cos=mean_cos, cos_vv=cos_vv, cos_tt=cos_tt,
        n_vis=int(vis_mask.sum()), n_txt=int((~vis_mask).sum()),
        correct=bool(res["correct"]), unextracted=res["prediction"] is None,
        category=str(row.get("category") or "unknown"),
    ))
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{N} done, mean_cos={mean_cos:.4f}", flush=True)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"saved -> {OUT_CSV}")

# ---------- 统计 ----------
import numpy as np
from scipy import stats

mc = np.array([r["mean_cos"] for r in rows])
y = np.array([r["correct"] for r in rows], dtype=float)
unext = np.array([r["unextracted"] for r in rows])

def report(tag, mc, y):
    r_pb, p_pb = stats.pointbiserialr(y, mc)
    rho, p_sp = stats.spearmanr(mc, y)
    g1, g0 = mc[y == 1], mc[y == 0]
    u, p_mw = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    auc = u / (len(g1) * len(g0))
    print(f"[{tag}] n={len(y)} (correct={int(y.sum())})")
    print(f"  mean_cos: correct={g1.mean():.4f}±{g1.std():.4f}  incorrect={g0.mean():.4f}±{g0.std():.4f}")
    print(f"  point-biserial r={r_pb:+.4f} (p={p_pb:.4f})   Spearman rho={rho:+.4f} (p={p_sp:.4f})")
    print(f"  Mann-Whitney AUC={auc:.4f} (p={p_mw:.4f})")

report("all", mc, y)
keep = ~unext
report("excl-unextracted", mc[keep], y[keep])

# 分 category 的点二列相关（探索性）
for cat in sorted(set(r["category"] for r in rows)):
    m = np.array([r["category"] == cat for r in rows])
    if m.sum() >= 50 and 0 < y[m].sum() < m.sum():
        r_pb, p_pb = stats.pointbiserialr(y[m], mc[m])
        print(f"  [{cat}] n={int(m.sum())} r={r_pb:+.4f} (p={p_pb:.4f})")
