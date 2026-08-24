"""MMLU 评测：SGLang（或 HF）上算 continuation logprob；可选每题后 online TTA（HF+PEFT）。"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _continuation_logprob_hf(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix: str,
    continuation: str,
    device: torch.device,
) -> float:
    """prefix + continuation 上，仅对 continuation 的 token 求和 log p（HF 回退）。"""
    full = prefix + continuation
    pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
    full_enc = tokenizer(full, return_tensors="pt", add_special_tokens=True)
    pref_ids = pref["input_ids"].to(device)
    full_ids = full_enc["input_ids"].to(device)
    attn = full_enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    plen = pref_ids.shape[1]
    if full_ids.shape[1] <= plen:
        return float("-inf")
    fwd = {"input_ids": full_ids, "attention_mask": attn} if attn is not None else {"input_ids": full_ids}
    out = model(**fwd)
    logits = out.logits[0].float()
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for t in range(plen - 1, full_ids.shape[1] - 1):
        tid = int(full_ids[0, t + 1].item())
        total += float(logp[t, tid].item())
    return total


def _score_choice(
    inference: Any,
    tokenizer: Any,
    prefix: str,
    continuation: str,
    *,
    hf_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> float:
    full = prefix + continuation
    pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
    plen = int(pref["input_ids"].shape[1])
    if hasattr(inference, "score_logprob_sum"):
        return float(inference.score_logprob_sum(full_text=full, prefix_len_tokens=plen))
    if hf_model is not None and device is not None:
        return _continuation_logprob_hf(hf_model, tokenizer, prefix, continuation, device)
    raise RuntimeError("无法计算 logprob：缺少 SGLang engine 或 HF 模型")


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_mmlu")
def main(cfg: DictConfig) -> None:
    from peft import PeftModel

    from openttl.adapters.registry import extract_model_cfg
    from openttl.data.mmlu import format_mmlu_prompt_no_answer, load_mmlu_source_dataset
    from openttl.models.loader import load_adapter
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.batching import build_tta_batch, strategy_to_label_mode

    hf_path = str(cfg.hf_path)
    hf_name = str(cfg.hf_name)
    split = str(cfg.split)
    max_s = OmegaConf.select(cfg, "max_samples")
    suffix = str(cfg.answer_suffix)

    mc = extract_model_cfg(cfg)
    adapter = load_adapter(cfg)
    adapter.load_processor(mc)
    tokenizer = adapter.tokenizer()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = str(OmegaConf.select(cfg, "inference.backend") or "sglang").lower()
    online_on = bool(OmegaConf.select(cfg, "online.enabled") or False)
    peft_on = bool(OmegaConf.select(cfg, "model.peft.enabled") or False)
    if online_on and bool(OmegaConf.select(cfg, "merge_lora") or False):
        raise ValueError("online.enabled 与 merge_lora 不能同时启用（merge 后无 PEFT 适配器可同步）")

    base = adapter.load_model(mc)
    ap = OmegaConf.select(cfg, "adapter_path")
    if ap:
        train_model = PeftModel.from_pretrained(base, ap)
    elif peft_on:
        train_model = inject_lora(base, cfg.model.peft)
    else:
        train_model = base

    if bool(OmegaConf.select(cfg, "merge_lora") or False) and hasattr(train_model, "merge_and_unload"):
        train_model = train_model.merge_and_unload()

    train_model.to(dev)

    initial_lora_path = None
    runner = None
    if online_on:
        if not peft_on and not ap:
            raise ValueError("online.enabled=true 需要 PEFT（model.peft.enabled 或 adapter_path）")
        from openttl.online.tta_runner import OnlineTTARunner

        initial_lora_path = str(OnlineTTARunner.initial_adapter_path(cfg, train_model, cfg.inference))

    infer = None
    hf_for_score: Optional[torch.nn.Module] = None

    if backend == "sglang":
        from openttl.inference.sglang_engine import build_sglang_engine_from_omegaconf

        infer = build_sglang_engine_from_omegaconf(
            cfg.model,
            cfg.inference,
            tokenizer,
            initial_lora_path if online_on else None,
        )
    elif backend == "hf":
        from openttl.inference.hf_engine import HuggingFaceEngine

        train_model.eval()
        infer = HuggingFaceEngine(train_model, adapter, dev)
        hf_for_score = train_model
    else:
        raise ValueError(f"未知 inference.backend: {backend}")

    if online_on:
        from openttl.online.tta_runner import OnlineTTARunner

        runner = OnlineTTARunner(
            cfg,
            model=train_model,
            adapter=adapter,
            inference=infer,
            device=dev,
        )

    ds = load_mmlu_source_dataset(cfg)
    if max_s is not None:
        ds = ds.select(range(min(int(max_s), len(ds))))

    correct = 0
    n = 0
    for row in ds:
        train_model.eval()
        prefix = format_mmlu_prompt_no_answer(row["question"], row["choices"])
        scores = []
        for letter in ("A", "B", "C", "D"):
            cont = f"{suffix}{letter}"
            scores.append(
                _score_choice(
                    infer,
                    tokenizer,
                    prefix,
                    cont,
                    hf_model=hf_for_score,
                    device=dev,
                )
            )
        pred = int(max(range(4), key=lambda i: scores[i]))
        gold = row["answer"]
        if isinstance(gold, str) and gold in "ABCD":
            gold_i = ord(gold) - ord("A")
        else:
            gold_i = int(gold)
        if pred == gold_i:
            correct += 1
        n += 1

        if runner is not None and runner.enabled():
            strat_name = str(OmegaConf.select(cfg, "strategy.name") or "tent").lower()
            max_len = int(OmegaConf.select(cfg, "online.max_length") or 512)
            lm = strategy_to_label_mode(strat_name, prompt_only_tta=True)
            batch = build_tta_batch(
                adapter,
                chat_prompt_text=prefix,
                prompt_plain=prefix,
                images=None,
                response=None,
                max_length=max_len,
                device=dev,
                label_mode=lm,
            )
            try:
                runner.update(batch)
            except Exception as e:
                print(f"Warning: TTA update failed: {e}")

    if backend == "sglang" and infer is not None:
        infer.shutdown()

    acc = correct / n if n else 0.0
    metrics = {
        "accuracy": acc,
        "correct": correct,
        "total": n,
        "hf_path": hf_path,
        "hf_name": hf_name,
        "split": split,
    }
    outp = Path(str(cfg.output_json))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
