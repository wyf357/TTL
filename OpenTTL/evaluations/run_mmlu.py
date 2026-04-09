"""MMLU 评测：仅在推理阶段使用标准答案计算准确率（不参与 TTA 梯度）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _continuation_logprob(
    model: torch.nn.Module,
    tokenizer,
    prefix: str,
    continuation: str,
    device: torch.device,
) -> float:
    """prefix + continuation 上，仅对 continuation 的 token 求和 log p。"""
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


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_mmlu")
def main(cfg: DictConfig) -> None:
    from peft import PeftModel

    from openttl.data.mmlu import format_mmlu_prompt_no_answer, load_mmlu_source_dataset
    from openttl.models.loader import load_causal_lm, load_tokenizer

    hf_path = str(cfg.hf_path)
    hf_name = str(cfg.hf_name)
    split = str(cfg.split)
    max_s = OmegaConf.select(cfg, "max_samples")
    suffix = str(cfg.answer_suffix)

    tokenizer = load_tokenizer(cfg.model)
    model = load_causal_lm(cfg.model)
    ap = OmegaConf.select(cfg, "adapter_path")
    if ap:
        model = PeftModel.from_pretrained(model, ap)
    if bool(OmegaConf.select(cfg, "merge_lora") or False) and hasattr(
        model, "merge_and_unload"
    ):
        model = model.merge_and_unload()
    model.eval()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    ds = load_mmlu_source_dataset(cfg)
    if max_s is not None:
        ds = ds.select(range(min(int(max_s), len(ds))))

    correct = 0
    n = 0
    for row in ds:
        prefix = format_mmlu_prompt_no_answer(row["question"], row["choices"])
        scores = []
        for letter in ("A", "B", "C", "D"):
            cont = f"{suffix}{letter}"
            scores.append(
                _continuation_logprob(model, tokenizer, prefix, cont, dev)
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
