"""Exp-1：观测假设（N=400，只前向，不生成）。

检验 r_j 动态范围、尖峰度 vs 对错、cosine top-k 与 gold saliency IoU。

用法:
  python verify_hyp_observe.py [N=400]
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
from scipy import stats

from openttl.adapters.qwen3_5 import Qwen3_5Adapter
from openttl.data.mmstar import load_mmstar_table
from openttl.strategies.hyp_verify import (
    DEFAULT_BASELINE_JSON,
    DEFAULT_MMSTAR_ROOT,
    DEFAULT_MODEL_PATH,
    HYP_OUTPUT_DIR,
    compute_gold_saliency,
    compute_relevance,
    default_model_cfg,
    prepare_sample_context,
    random_iou_baseline,
    spikiness_metrics,
    topk_iou,
)
from openttl.strategies.mma import capture_layer_hidden

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT_CSV = HYP_OUTPUT_DIR / "observe.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

baseline = json.load(open(DEFAULT_BASELINE_JSON))["results"]
cfg = default_model_cfg(DEFAULT_MODEL_PATH)
adapter = Qwen3_5Adapter()
adapter.load_processor(cfg)
model = adapter.load_model(cfg).to(device).eval()
tok = adapter.tokenizer()

ds = load_mmstar_table(local_root=DEFAULT_MMSTAR_ROOT)
print(f"Exp-1 observe: N={N}, l* scoring", flush=True)

HYP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rows = []
rand_iou10 = random_iou_baseline(192, 10)
rand_iou20 = random_iou_baseline(192, 20)

for i in range(N):
    row = ds[i]
    res = baseline[i]
    question = str(row["question"])
    ctx = prepare_sample_context(
        model, adapter, question=question, image=row["image"], device=device
    )
    h = capture_layer_hidden(model, ctx.enc, ctx.score_layer)[0]

    r_q = compute_relevance(h, ctx.vis_mask, ctx.q_mask, mode="max")
    r_all = compute_relevance(h, ctx.vis_mask, ctx.all_txt_mask, mode="max")

    sp_q = spikiness_metrics(r_q)
    sp_all = spikiness_metrics(r_all)

    sal = compute_gold_saliency(
        model,
        ctx.enc,
        ctx.score_layer,
        ctx.vis_mask,
        str(row["answer"]).strip().upper(),
        tok,
    )
    iou10 = iou20 = float("nan")
    if sal is not None and sal.numel() > 0:
        iou10 = topk_iou(r_q.cpu(), sal, 10)
        iou20 = topk_iou(r_q.cpu(), sal, 20)

    rows.append(
        dict(
            idx=i,
            n_vis=int(ctx.vis_mask.sum()),
            n_q=int(ctx.q_mask.sum()),
            r_mean=float(r_q.mean().item()),
            r_std=float(r_q.std().item()),
            r_cv=float(r_q.std().item() / (r_q.mean().item() + 1e-12)),
            entropy_q=sp_q["entropy"],
            entropy_all=sp_all["entropy"],
            peak_gap_q=sp_q["peak_gap"],
            top10_mass_q=sp_q["top10_mass"],
            iou10=iou10,
            iou20=iou20,
            correct=bool(res["correct"]),
            unextracted=res.get("prediction") is None,
            category=str(row.get("category") or "unknown"),
        )
    )
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N}  r_cv={rows[-1]['r_cv']:.4f}  iou10={iou10:.3f}", flush=True)

    del h, r_q, r_all, ctx, sal
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"saved -> {OUT_CSV}")


def report(tag: str, x: np.ndarray, y: np.ndarray) -> None:
    if len(np.unique(y)) < 2:
        print(f"[{tag}] skipped (single class)")
        return
    r_pb, p_pb = stats.pointbiserialr(y, x)
    rho, p_sp = stats.spearmanr(x, y)
    g1, g0 = x[y == 1], x[y == 0]
    u, p_mw = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    auc = u / (len(g1) * len(g0))
    print(f"[{tag}] n={len(y)} correct={int(y.sum())}")
    print(f"  correct={g1.mean():.4f}±{g1.std():.4f}  incorrect={g0.mean():.4f}±{g0.std():.4f}")
    print(f"  r={r_pb:+.4f} (p={p_pb:.4f})  rho={rho:+.4f} (p={p_sp:.4f})  AUC={auc:.4f}")


y = np.array([r["correct"] for r in rows], dtype=float)
unext = np.array([r["unextracted"] for r in rows], dtype=bool)

print("\n" + "=" * 60)
print("r_j dynamics (question-conditioned)")
print(f"  mean CV = {np.mean([r['r_cv'] for r in rows]):.4f}  (MMA mean-cos CV ~0.08)")
print(f"  mean entropy_q = {np.mean([r['entropy_q'] for r in rows]):.4f}")
print(f"  mean entropy_all = {np.mean([r['entropy_all'] for r in rows]):.4f}")

iou10 = np.array([r["iou10"] for r in rows])
iou20 = np.array([r["iou20"] for r in rows])
valid = ~np.isnan(iou10)
print(f"\nCosine–saliency IoU (n={valid.sum()})")
print(f"  IoU@10 mean={iou10[valid].mean():.4f}  random~{rand_iou10:.4f}")
print(f"  IoU@20 mean={iou20[valid].mean():.4f}  random~{rand_iou20:.4f}")

print("\nCorrelations with correctness:")
report("peak_gap_q", np.array([r["peak_gap_q"] for r in rows]), y)
report("entropy_q", np.array([r["entropy_q"] for r in rows]), y)
report("iou10", iou10[valid], y[valid])

keep = ~unext
report("peak_gap_q excl-unext", np.array([r["peak_gap_q"] for r in rows])[keep], y[keep])
report("iou10 excl-unext", iou10[valid & keep], y[valid & keep])

summary = {
    "N": N,
    "mean_r_cv": float(np.mean([r["r_cv"] for r in rows])),
    "mean_iou10": float(iou10[valid].mean()) if valid.any() else None,
    "random_iou10": rand_iou10,
    "mean_entropy_q": float(np.mean([r["entropy_q"] for r in rows])),
    "mean_entropy_all": float(np.mean([r["entropy_all"] for r in rows])),
}
(HYP_OUTPUT_DIR / "observe_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSummary -> {HYP_OUTPUT_DIR / 'observe_summary.json'}")
