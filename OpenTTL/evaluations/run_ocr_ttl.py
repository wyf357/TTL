"""OCRBench / OmniDocBench 评测：HF 后端逐样本推理 + 可选 online TTA（tent/tlm/come）。

协议与 run_mmbench_cn.py 一致：先用当前权重生成并判分，再用该样本做一步 TTA
更新（不接触标签）。判分直接调用 lmms-eval 任务函数，口径与 lmms-eval 完全一致。

用法：
  python evaluations/run_ocr_ttl.py benchmark=ocrbench online.enabled=false model.peft.enabled=false
  python evaluations/run_ocr_ttl.py benchmark=omnidocbench strategy=come online.enabled=true ...
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _tta_step(
    runner: Any,
    adapter: Any,
    cfg: DictConfig,
    *,
    chat_prompt_text: str,
    query_plain: str,
    images: List[Any],
    response: str,
    device: torch.device,
) -> float:
    from openttl.online.batching import (
        build_tta_batch,
        strategy_suppresses_response,
        strategy_to_label_mode,
    )

    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "tent").lower()
    lm = strategy_to_label_mode(strat_name, prompt_only_tta=False)
    eff_response = None if strategy_suppresses_response(strat_name) else response
    max_tok = int(OmegaConf.select(cfg, "online.max_length") or 4096)

    batch = build_tta_batch(
        adapter,
        chat_prompt_text=chat_prompt_text,
        prompt_plain=query_plain,
        images=images,
        response=eff_response,
        max_length=max_tok,
        device=device,
        label_mode=lm,
        enable_thinking=False,
        mm_encode_like_inference=True,
    )
    return runner.update(batch)


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_ocr_ttl")
def main(cfg: DictConfig) -> None:
    from openttl.adapters.registry import extract_model_cfg
    from openttl.models.loader import load_adapter
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.tta_runner import OnlineTTARunner

    benchmark = str(cfg.benchmark).lower()
    if benchmark not in ("ocrbench", "omnidocbench"):
        raise ValueError(f"未知 benchmark: {benchmark}")

    if benchmark == "ocrbench":
        from lmms_eval.tasks.ocrbench.utils import ocrbench_process_results as process_results
        from openttl.data.ocr_bench import load_ocrbench as load_ds, ocrbench_prompt as build_prompt
        gen_max_default = 128  # 对齐 ocrbench.yaml generation_kwargs
    else:
        from lmms_eval.tasks.omnidocbench.utils import (
            omnidocbench_aggregate_formula_edit,
            omnidocbench_aggregate_overall,
            omnidocbench_aggregate_table_teds,
            omnidocbench_aggregate_text_edit,
            omnidocbench_process_results as process_results,
        )
        from openttl.data.ocr_bench import (
            decode_omni_image,
            load_omnidocbench as load_ds,
            omnidocbench_prompt as build_prompt,
        )
        gen_max_default = 4096  # 对齐 omnidocbench.yaml generation_kwargs

    max_samples = OmegaConf.select(cfg, "max_samples")
    output_json = Path(str(cfg.output_json))
    output_jsonl = output_json.with_suffix(".jsonl")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    mc = extract_model_cfg(cfg)
    adapter = load_adapter(cfg)
    adapter.load_processor(mc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online_on = bool(OmegaConf.select(cfg, "online.enabled") or False)
    peft_on = bool(OmegaConf.select(cfg, "model.peft.enabled") or False)
    if online_on and not peft_on:
        raise ValueError("online.enabled=true 需要 model.peft.enabled=true")

    train_model = adapter.load_model(mc)
    if peft_on:
        train_model = inject_lora(train_model, cfg.model.peft)
    train_model.to(device)

    from openttl.inference.hf_engine import HuggingFaceEngine

    train_model.eval()
    infer = HuggingFaceEngine(train_model, adapter, device)

    runner: Optional[Any] = None
    if online_on:
        runner = OnlineTTARunner(cfg, model=train_model, adapter=adapter, inference=infer, device=device)

    ds = load_ds()
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    # 断点续跑
    done_ids = set()
    if output_jsonl.exists():
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["rid"])
                except Exception:
                    pass
        print(f"Resume: {len(done_ids)} samples already done, skipping.")

    gen_max = int(OmegaConf.select(cfg, "gen_max_new_tokens") or gen_max_default)
    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "none").lower() if online_on else "none"

    print(f"Benchmark: {benchmark}, n={len(ds)}, gen_max={gen_max}")
    print(f"Model: {mc.pretrained_model_name_or_path}")
    print(f"Strategy: {strat_name} (online.enabled={online_on})")

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    fout = open(output_jsonl, "a", encoding="utf-8")

    for i, row in enumerate(ds):
        rid = int(row["index"]) if benchmark == "omnidocbench" else i
        if rid in done_ids:
            continue

        if benchmark == "omnidocbench":
            from openttl.data.ocr_bench import decode_omni_image

            image = decode_omni_image(row["image"])
        else:
            image = row["image"].convert("RGB")
        query = build_prompt(row)

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": query}],
            }
        ]
        chat_prompt_text = adapter.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        response = ""
        tta_loss: Optional[float] = None
        err: Optional[str] = None
        try:
            raw = infer.generate(
                chat_prompt_text,
                image_data=[image],
                sampling_params={"max_new_tokens": gen_max, "temperature": 0.0, "top_p": 1.0},
            )
            response = str(raw).strip()

            if runner is not None and runner.enabled():
                try:
                    tta_loss = _tta_step(
                        runner, adapter, cfg,
                        chat_prompt_text=chat_prompt_text, query_plain=query,
                        images=[image], response=response, device=device,
                    )
                except Exception as te:
                    print(f"Warning: TTA update failed at sample {i} (rid={rid}): {te}")
        except Exception as e:
            err = str(e)
            print(f"Error processing sample {i} (rid={rid}): {e}")

        # 判分（lmms-eval 任务函数，口径一致）
        score_payload: Optional[Dict[str, Any]] = None
        try:
            if benchmark == "ocrbench":
                score_payload = process_results(row, [response])["ocrbench_accuracy"]
            else:
                score_payload = process_results(row, [response])["omnidocbench_overall"]
        except Exception as se:
            print(f"Warning: scoring failed at sample {i} (rid={rid}): {se}")

        rec = {
            "rid": rid,
            "response": response[:2000],
            "score": score_payload,
            "tta_loss": tta_loss,
            "error": err,
        }
        if benchmark == "ocrbench":
            rec["question_type"] = row.get("question_type")
        records.append(rec)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

        n_done = len(records)
        if n_done % 20 == 0 or n_done == 1:
            dt = (time.time() - t0) / n_done
            if benchmark == "ocrbench":
                acc = sum(r["score"]["score"] for r in records if r["score"]) / max(
                    1, sum(1 for r in records if r["score"])
                )
                print(f"[{n_done}/{len(ds) - len(done_ids)}] acc={acc:.4f} loss={tta_loss} ({dt:.1f}s/sample)")
            else:
                ov = [r["score"]["overall"] for r in records if r["score"]]
                print(f"[{n_done}/{len(ds) - len(done_ids)}] overall={sum(ov)/max(1,len(ov)):.2f} loss={tta_loss} ({dt:.1f}s/sample)")

        del image

    fout.close()

    # 汇总
    if benchmark == "ocrbench":
        scored = [r for r in records if r["score"]]
        total_score = sum(r["score"]["score"] for r in scored)
        types: Dict[str, List[int]] = {}
        for r in scored:
            types.setdefault(str(r["question_type"]), []).append(r["score"]["score"])
        metrics: Dict[str, Any] = {
            "overall_acc": total_score / len(scored) if scored else 0.0,
            "total_score": total_score,
            "total": len(scored),
            "question_type_acc": {k: sum(v) / len(v) for k, v in sorted(types.items())},
        }
    else:
        payloads = [r["score"] for r in records if r["score"]]
        metrics = {
            "overall": omnidocbench_aggregate_overall(payloads),
            "text_edit": omnidocbench_aggregate_text_edit(payloads),
            "table_teds": omnidocbench_aggregate_table_teds(payloads),
            "formula_edit": omnidocbench_aggregate_formula_edit(payloads),
            "total": len(payloads),
        }

    metrics.update(
        {
            "benchmark": benchmark,
            "strategy": strat_name,
            "online_enabled": online_on,
            "model": str(mc.pretrained_model_name_or_path),
            "gen_max_new_tokens": gen_max,
            "elapsed_sec": round(time.time() - t0, 1),
        }
    )

    output_json.write_text(
        json.dumps({"metrics": metrics, "config": OmegaConf.to_container(cfg, resolve=True)},
                   ensure_ascii=False, indent=2)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Results saved to: {output_json} (+ {output_jsonl})")


if __name__ == "__main__":
    main()
