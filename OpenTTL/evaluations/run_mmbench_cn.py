"""MMBench-CN 评测：HF 后端逐样本推理；可选每题后 online TTA（HF+PEFT，tent/eata/come/tlm）。

协议（与 run_erqa.py 一致）：先用当前权重生成答案并判分，再用该样本做一步 TTA
更新（不接触 gold 标签），权重随样本流式累积。

用法：
  python evaluations/run_mmbench_cn.py strategy=tent online.enabled=true model.peft.enabled=true
  python evaluations/run_mmbench_cn.py online.enabled=false   # 无 TTA 基线
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
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

_THINK_CLOSE = "</think>"


def _strip_thinking_block(text: str) -> str:
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE)[-1].strip()
    return text


def extract_answer_letter(response: str) -> Optional[str]:
    """从模型输出中抽取选项字母（A-E）。"""
    answer_part = _strip_thinking_block(response)
    answer_part = re.sub(r"<\|[^|]+\|>", "", answer_part).strip()

    patterns = [
        r"^([A-E])\b",                       # 直接以字母开头（post_prompt 要求的行为）
        r"([A-E])\s*[.。:：]",               # "B." / "B。"
        r"[Aa]nswer\s*[:：]\s*([A-E])",
        r"答案\s*(?:是|为)?\s*[:：]?\s*([A-E])",
        r"选项\s*[:：]?\s*([A-E])",
        r"^([A-E])$",
    ]
    for pat in patterns:
        m = re.search(pat, answer_part, re.MULTILINE)
        if m:
            return m.group(1).upper()

    first = answer_part[:20]
    for letter in "ABCDE":
        if letter in first:
            return letter
    return None


def _tta_step(
    runner: Any,
    adapter: Any,
    cfg: DictConfig,
    *,
    chat_prompt_text: str,
    query_plain: str,
    images: Optional[List[Any]],
    response: str,
    device: torch.device,
) -> float:
    """单步 online TTA；只使用 prompt 与模型自身输出，不接触 gold。"""
    from openttl.online.batching import (
        build_tta_batch,
        strategy_suppresses_response,
        strategy_to_label_mode,
    )

    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "tent").lower()
    lm = strategy_to_label_mode(strat_name, prompt_only_tta=False)
    eff_response = None if strategy_suppresses_response(strat_name) else response
    max_tok = int(OmegaConf.select(cfg, "online.max_length") or 4096)
    enable_thinking = bool(OmegaConf.select(cfg, "mmbench.enable_thinking") or False)

    batch = build_tta_batch(
        adapter,
        chat_prompt_text=chat_prompt_text,
        prompt_plain=query_plain,
        images=images,
        response=eff_response,
        max_length=max_tok,
        device=device,
        label_mode=lm,
        enable_thinking=enable_thinking,
        # 与 generate 使用同一 chat_prompt_text（add_generation_prompt=True）
        mm_encode_like_inference=True,
    )
    return runner.update(batch)


def _category_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按 category / L2-category 汇总准确率（对齐 lmms-eval 输出结构）。"""
    def _agg(key: str) -> Dict[str, float]:
        buckets: Dict[str, List[bool]] = {}
        for r in records:
            buckets.setdefault(str(r.get(key) or "unknown"), []).append(bool(r["correct"]))
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}

    return {
        "category_acc": _agg("category"),
        "l2_category_acc": _agg("l2_category"),
    }


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_mmbench_cn")
def main(cfg: DictConfig) -> None:
    from openttl.adapters.registry import extract_model_cfg
    from openttl.data.mmbench_cn import format_mmbench_cn_query, load_mmbench_cn_dataset
    from openttl.models.loader import load_adapter
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.tta_runner import OnlineTTARunner

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
    backend = str(OmegaConf.select(cfg, "inference.backend") or "hf").lower()
    if backend != "hf":
        raise ValueError("本脚本仅支持 inference.backend=hf（TTA 与推理共用同一进程内模型）")

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

    ds = load_mmbench_cn_dataset(
        str(cfg.dataset_path), str(cfg.dataset_name), str(cfg.split)
    )
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    # 断点续跑：跳过 JSONL 中已有的 index
    done_indices = set()
    if output_jsonl.exists():
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["index"])
                except Exception:
                    pass
        print(f"Resume: {len(done_indices)} samples already done, skipping.")

    enable_thinking = bool(OmegaConf.select(cfg, "mmbench.enable_thinking") or False)
    gen_max = int(OmegaConf.select(cfg, "mmbench.max_new_tokens") or 256)
    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "none").lower() if online_on else "none"

    print(f"Model: {mc.pretrained_model_name_or_path}")
    print(f"Strategy: {strat_name} (online.enabled={online_on}), backend=hf, enable_thinking={enable_thinking}")
    print(f"Dataset: {cfg.dataset_path} {cfg.dataset_name} {cfg.split}, n={len(ds)}")

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    fout = open(output_jsonl, "a", encoding="utf-8")

    for i, row in enumerate(ds):
        index = int(row["index"])
        if index in done_indices:
            continue

        image = row["image"].convert("RGB")
        query = format_mmbench_cn_query(row)
        gold = str(row["answer"]).strip().upper()

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": query}],
            }
        ]
        chat_prompt_text = adapter.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        pred_letter: Optional[str] = None
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
            pred_letter = extract_answer_letter(response)

            if runner is not None and runner.enabled():
                try:
                    tta_loss = _tta_step(
                        runner,
                        adapter,
                        cfg,
                        chat_prompt_text=chat_prompt_text,
                        query_plain=query,
                        images=[image],
                        response=response,
                        device=device,
                    )
                except Exception as te:
                    print(f"Warning: TTA update failed at sample {i} (index={index}): {te}")
        except Exception as e:
            err = str(e)
            print(f"Error processing sample {i} (index={index}): {e}")

        is_correct = bool(pred_letter) and pred_letter == gold
        rec = {
            "index": index,
            "category": row["category"],
            "l2_category": row["L2-category"],
            "answer": gold,
            "prediction": pred_letter,
            "response": response[:500],
            "correct": is_correct,
            "tta_loss": tta_loss,
            "error": err,
        }
        records.append(rec)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

        n_done = len(records)
        if n_done % 20 == 0 or n_done == 1:
            acc = sum(r["correct"] for r in records) / n_done
            dt = (time.time() - t0) / n_done
            print(f"[{n_done}/{len(ds) - len(done_indices)}] acc={acc:.4f} "
                  f"pred={pred_letter} gold={gold} loss={tta_loss} ({dt:.1f}s/sample)")

        del image

    fout.close()

    correct = sum(r["correct"] for r in records)
    total = len(records)
    metrics: Dict[str, Any] = {
        "overall_acc": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "strategy": strat_name,
        "online_enabled": online_on,
        "model": str(mc.pretrained_model_name_or_path),
        "dataset": f"{cfg.dataset_path}/{cfg.dataset_name}/{cfg.split}",
        "enable_thinking": enable_thinking,
        "max_new_tokens": gen_max,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    metrics.update(_category_metrics(records))

    output_json.write_text(
        json.dumps({"metrics": metrics, "config": OmegaConf.to_container(cfg, resolve=True)},
                   ensure_ascii=False, indent=2)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Results saved to: {output_json} (+ {output_jsonl})")


if __name__ == "__main__":
    main()
