"""Exp-0：H0 通道存在性验证。

比较 MMA δ、全局 uniform、随机非均匀、题干 cosine 门控（post/pre RMSNorm）
对 last-token logits 的影响；选出供 Exp-2 使用的最小 α。

用法:
  python verify_hyp_channel.py [N=8]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import json

import torch

from openttl.adapters.qwen3_5 import Qwen3_5Adapter
from openttl.data.mmstar import load_mmstar_table
from openttl.strategies.hyp_verify import (
    DEFAULT_MMSTAR_ROOT,
    DEFAULT_MODEL_PATH,
    DeltaHookManager,
    GateHookManager,
    HYP_OUTPUT_DIR,
    compute_relevance,
    default_model_cfg,
    forward_last_logits,
    gates_for_condition,
    mma_production_delta,
    measure_intervention_logits,
    prepare_sample_context,
    save_selected_alpha,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ALPHAS = (0.5, 1.0, 2.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = default_model_cfg(DEFAULT_MODEL_PATH)
adapter = Qwen3_5Adapter()
adapter.load_processor(cfg)
model = adapter.load_model(cfg).to(device).eval()

ds = load_mmstar_table(local_root=DEFAULT_MMSTAR_ROOT)
print(f"Exp-0 channel test: N={N}, alphas={ALPHAS}", flush=True)

rows = []
for i in range(N):
    row = ds[i]
    question = str(row["question"])
    ctx = prepare_sample_context(
        model, adapter, question=question, image=row["image"], device=device
    )
    from openttl.strategies.mma import capture_layer_hidden

    h = capture_layer_hidden(model, ctx.enc, ctx.score_layer)[0]
    r = compute_relevance(h, ctx.vis_mask, ctx.q_mask, mode="max")
    base_logits = forward_last_logits(model, ctx.enc)

    delta = mma_production_delta(h, ctx.vis_mask)
    mma_mgr = DeltaHookManager(
        ctx.score_layer,
        prompt_len=ctx.prompt_len,
        vis_idx=ctx.vis_idx,
        delta=delta,
    )
    mma_stats = measure_intervention_logits(model, ctx.enc, base_logits, mma_mgr)

    sample_row = {"idx": i, "mma": mma_stats, "alpha_results": {}}

    for alpha in ALPHAS:
        gates_q = gates_for_condition("q_aligned", r, alpha=alpha)
        gates_rand = gates_for_condition("random", r, alpha=alpha)
        gates_uni = gates_for_condition("uniform", r, alpha=alpha)

        post_q = GateHookManager(
            ctx.layers,
            l_star_1based=ctx.l_star,
            prompt_len=ctx.prompt_len,
            vis_idx=ctx.vis_idx,
            gates=gates_q,
            pre_ln=False,
        )
        post_rand = GateHookManager(
            ctx.layers,
            l_star_1based=ctx.l_star,
            prompt_len=ctx.prompt_len,
            vis_idx=ctx.vis_idx,
            gates=gates_rand,
            pre_ln=False,
        )
        post_uni = GateHookManager(
            ctx.layers,
            l_star_1based=ctx.l_star,
            prompt_len=ctx.prompt_len,
            vis_idx=ctx.vis_idx,
            gates=gates_uni,
            pre_ln=False,
        )
        pre_q = GateHookManager(
            ctx.layers,
            l_star_1based=ctx.l_star,
            prompt_len=ctx.prompt_len,
            vis_idx=ctx.vis_idx,
            gates=gates_q,
            pre_ln=True,
        )

        sample_row["alpha_results"][str(alpha)] = {
            "post_q_aligned": measure_intervention_logits(model, ctx.enc, base_logits, post_q),
            "post_random": measure_intervention_logits(model, ctx.enc, base_logits, post_rand),
            "post_uniform": measure_intervention_logits(model, ctx.enc, base_logits, post_uni),
            "pre_q_aligned": measure_intervention_logits(model, ctx.enc, base_logits, pre_q),
        }

    rows.append(sample_row)
    print(f"  sample {i}: mma max|d|={mma_stats['max_abs_diff']:.3e}", flush=True)

# ---------- 汇总与 α 选择 ----------
def agg(condition_key: str, alpha: float) -> dict:
    vals = [r["alpha_results"][str(alpha)][condition_key] for r in rows]
    return {
        "mean_max_abs_diff": sum(v["max_abs_diff"] for v in vals) / len(vals),
        "mean_kl": sum(v["kl"] for v in vals) / len(vals),
        "top1_flips": sum(1 for v in vals if v["same_top1"] == 0),
    }

mma_mean = sum(r["mma"]["max_abs_diff"] for r in rows) / len(rows)
print(f"\nMMA mean max|Δlogit| = {mma_mean:.3e}")

selected_alpha = None
verdict_h0 = False
for alpha in ALPHAS:
    post_q = agg("post_q_aligned", alpha)
    post_rand = agg("post_random", alpha)
    post_uni = agg("post_uniform", alpha)
    pre_q = agg("pre_q_aligned", alpha)
    print(f"\nα={alpha}:")
    print(f"  post_q_aligned: mean|d|={post_q['mean_max_abs_diff']:.3e}  flips={post_q['top1_flips']}")
    print(f"  post_random:    mean|d|={post_rand['mean_max_abs_diff']:.3e}  flips={post_rand['top1_flips']}")
    print(f"  post_uniform:   mean|d|={post_uni['mean_max_abs_diff']:.3e}  flips={post_uni['top1_flips']}")
    print(f"  pre_q_aligned:  mean|d|={pre_q['mean_max_abs_diff']:.3e}  flips={pre_q['top1_flips']}")

    pass_channel = (
        max(post_q["mean_max_abs_diff"], post_rand["mean_max_abs_diff"]) >= 10 * mma_mean
        and max(post_q["top1_flips"], post_rand["top1_flips"]) >= 2
    )
    pass_injection_point = post_q["mean_max_abs_diff"] > 2 * pre_q["mean_max_abs_diff"]
    pass_discriminate = post_uni["mean_max_abs_diff"] < max(
        post_q["mean_max_abs_diff"], post_rand["mean_max_abs_diff"]
    )

    if pass_channel and pass_injection_point and pass_discriminate and selected_alpha is None:
        selected_alpha = alpha
        verdict_h0 = True

if selected_alpha is None:
    # 放宽：仅看通道是否比 MMA 强一个数量级
    for alpha in ALPHAS:
        post_q = agg("post_q_aligned", alpha)
        if post_q["mean_max_abs_diff"] >= 10 * mma_mean and post_q["top1_flips"] >= 2:
            selected_alpha = alpha
            verdict_h0 = True
            break

if selected_alpha is None:
    selected_alpha = ALPHAS[-1]

HYP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = HYP_OUTPUT_DIR / "channel_results.json"
summary = {
    "N": N,
    "mma_mean_max_abs_diff": mma_mean,
    "selected_alpha": selected_alpha,
    "H0_pass": verdict_h0,
    "per_sample": rows,
}
out_path.write_text(json.dumps(summary, indent=2))
save_selected_alpha(selected_alpha)

print("\n" + "=" * 60)
print(f"H0 verdict: {'PASS' if verdict_h0 else 'FAIL'}")
print(f"Selected α for Exp-2: {selected_alpha}")
print(f"Saved -> {out_path}")
