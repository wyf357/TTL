"""Exp-2：因果四路干预（N=200，生成答案）——主判决 H1/H2。

条件: baseline / Q-aligned / Q-anti / shuffle / uniform / oracle(可选)

用法:
  python verify_hyp_causal.py [N=200] [--oracle]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import torch

from openttl.adapters.qwen3_5 import Qwen3_5Adapter
from openttl.data.mmstar import load_mmstar_table
from openttl.eval.answer_extraction import extract_answer_letter
from openttl.strategies.hyp_verify import (
    DEFAULT_MMSTAR_ROOT,
    DEFAULT_MODEL_PATH,
    HYP_OUTPUT_DIR,
    compute_gold_saliency,
    compute_relevance,
    default_model_cfg,
    gates_for_condition,
    generate_with_gates,
    load_selected_alpha,
    paired_mcnemar,
    prepare_sample_context,
)
from openttl.strategies.mma import capture_layer_hidden

parser = argparse.ArgumentParser()
parser.add_argument("N", nargs="?", type=int, default=200)
parser.add_argument("--oracle", action="store_true", help="加 gold-saliency oracle 上界")
args = parser.parse_args()

N = args.N
CONDITIONS = ["baseline", "q_aligned", "q_anti", "shuffle", "uniform"]
if args.oracle:
    CONDITIONS.append("oracle")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = load_selected_alpha()
print(f"Exp-2 causal: N={N}, α={alpha}, conditions={CONDITIONS}", flush=True)

cfg = default_model_cfg(DEFAULT_MODEL_PATH)
adapter = Qwen3_5Adapter()
adapter.load_processor(cfg)
model = adapter.load_model(cfg).to(device).eval()
tok = adapter.tokenizer()

ds = load_mmstar_table(local_root=DEFAULT_MMSTAR_ROOT)
HYP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = HYP_OUTPUT_DIR / "causal.csv"

rows = []
correct_by_cond = {c: 0 for c in CONDITIONS}
total = 0

for i in range(N):
    row = ds[i]
    question = str(row["question"])
    gold = str(row["answer"]).strip().upper()
    ctx = prepare_sample_context(
        model, adapter, question=question, image=row["image"], device=device
    )
    h = capture_layer_hidden(model, ctx.enc, ctx.score_layer)[0]
    r = compute_relevance(h, ctx.vis_mask, ctx.q_mask, mode="max")
    sal = None
    if "oracle" in CONDITIONS:
        sal = compute_gold_saliency(
            model, ctx.enc, ctx.score_layer, ctx.vis_mask, gold, tok
        )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(i + 42)

    sample_preds: dict = {}
    for cond in CONDITIONS:
        if cond == "baseline":
            with torch.no_grad():
                pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
                gen_ids = model.generate(
                    **ctx.enc,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            text = adapter.decode_new_tokens(gen_ids, ctx.prompt_len).strip()
        else:
            gates = gates_for_condition(
                cond,
                r,
                alpha=alpha,
                generator=gen,
                saliency=sal,
            )
            text = generate_with_gates(model, adapter, ctx, gates, max_new_tokens=1024)

        pred = extract_answer_letter(text)
        ok = pred == gold if pred else False
        sample_preds[cond] = dict(pred=pred, correct=ok, text_head=text[:120])
        if ok:
            correct_by_cond[cond] += 1

    total += 1
    rows.append(
        dict(
            idx=i,
            gold=gold,
            **{f"{c}_pred": sample_preds[c]["pred"] for c in CONDITIONS},
            **{f"{c}_correct": sample_preds[c]["correct"] for c in CONDITIONS},
        )
    )
    if (i + 1) % 20 == 0:
        acc = {c: correct_by_cond[c] / total for c in CONDITIONS}
        print(f"  {i+1}/{N}  acc baseline={acc['baseline']:.3f} q={acc['q_aligned']:.3f}", flush=True)

    del h, r, ctx, sal, sample_preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

fieldnames = ["idx", "gold"] + [f"{c}_pred" for c in CONDITIONS] + [f"{c}_correct" for c in CONDITIONS]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

# ---------- 汇总 ----------
acc = {c: correct_by_cond[c] / total for c in CONDITIONS}
summary = {"N": total, "alpha": alpha, "accuracy": acc, "paired_tests": {}}

baseline_ok = [r["baseline_correct"] for r in rows]
for cond in CONDITIONS:
    if cond == "baseline":
        continue
    summary["paired_tests"][f"{cond}_vs_baseline"] = paired_mcnemar(
        baseline_ok, [r[f"{cond}_correct"] for r in rows]
    )
summary["paired_tests"]["q_aligned_vs_shuffle"] = paired_mcnemar(
    [r["shuffle_correct"] for r in rows],
    [r["q_aligned_correct"] for r in rows],
)

# H1/H2 判决
h1_support = (
    acc["q_aligned"] > acc["baseline"] + 0.005
    or (args.oracle and acc.get("oracle", 0) > acc["baseline"] + 0.01)
)
h2_support = (
    acc["q_aligned"] > acc["shuffle"] + 0.005
    and acc["q_anti"] < acc["baseline"] - 0.005
)
only_nonuniform = acc["shuffle"] > acc["baseline"] + 0.005 and abs(acc["q_aligned"] - acc["shuffle"]) < 0.01
only_global = abs(acc["uniform"] - acc["q_aligned"]) < 0.01 and acc["q_aligned"] <= acc["baseline"] + 0.005

if args.oracle and acc.get("oracle", 0) <= acc["baseline"] + 0.005:
    verdict = "H1_FAIL: oracle 也不涨，瓶颈不在看哪块"
elif only_global:
    verdict = "H2_FAIL: 仅全局加强，无问题条件化"
elif only_nonuniform:
    verdict = "H2_FAIL: 非均匀有用但问题条件化无效"
elif h2_support:
    verdict = "H1_H2_PASS: Q-aligned 优于 shuffle，Q-anti 变差"
elif h1_support:
    verdict = "H1_PARTIAL: 有增益但问题条件化证据不足"
else:
    verdict = "NO_EFFECT: 全无显著差异"

summary["verdict"] = verdict
summary["H1_support"] = h1_support
summary["H2_support"] = h2_support

summary_path = HYP_OUTPUT_DIR / "causal_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 60)
print("Causal accuracy:")
for c in CONDITIONS:
    print(f"  {c:12s}: {acc[c]:.4f} ({correct_by_cond[c]}/{total})")
print("\nPaired vs baseline (McNemar):")
for k, v in summary["paired_tests"].items():
    print(f"  {k}: Δacc={v['delta_acc']:+.4f}  p={v['p_value']:.4f}")
print(f"\nVerdict: {verdict}")
print(f"Saved -> {OUT_CSV}")
print(f"Saved -> {summary_path}")
