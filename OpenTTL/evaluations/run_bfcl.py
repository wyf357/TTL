"""BFCL (Berkeley Function Calling Leaderboard) 评测：预训练模型函数调用预测 + AST 评分。

数据与评分说明见 src/openttl/data/bfcl.py 与 src/openttl/eval/bfcl_eval.py。
产物：
  - output_json: 指标（类别、样本数、准确率等）
  - result_jsonl: 逐样本预测（{"id":..., "result":...}，兼容官方 answer 格式）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"


def _strip_thinking(text: str) -> str:
    """去掉 Qwen3.5 thinking 段，保留最终函数调用。"""
    if not text:
        return text
    o, c = _THINK_OPEN, _THINK_CLOSE
    prev = None
    while prev != text:
        prev = text
        text = re.sub(re.escape(o) + r".*?" + re.escape(c), "", text, flags=re.DOTALL)
    text = text.strip()
    if c in text:
        tail = text.split(c)[-1].strip()
        if tail:
            text = tail
    return text.strip()


def _build_prompt(tokenizer, row: dict, use_chat_template: bool) -> str:
    from openttl.data.bfcl import build_bfcl_messages, build_bfcl_plain_prompt

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tokenizer.apply_chat_template(
                build_bfcl_messages(row), enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(build_bfcl_messages(row), **kwargs)
    return build_bfcl_plain_prompt(row)


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_bfcl")
def main(cfg: DictConfig) -> None:
    from peft import PeftModel

    from openttl.data.bfcl import load_bfcl_category
    from openttl.eval.bfcl_eval import evaluate_bfcl_sample, summarize_bfcl_results
    from openttl.models.loader import load_causal_lm, load_tokenizer

    category = str(cfg.category)
    max_s = OmegaConf.select(cfg, "max_samples")
    max_new_tokens = int(OmegaConf.select(cfg, "max_new_tokens") or 256)
    use_chat_template = bool(OmegaConf.select(cfg, "use_chat_template") if OmegaConf.select(cfg, "use_chat_template") is not None else True)

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

    if getattr(model, "hf_device_map", None) is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
    else:
        dev = next(model.parameters()).device

    rows = load_bfcl_category(cfg)
    if max_s is not None:
        rows = rows[: int(max_s)]

    resp = Path(str(cfg.result_jsonl))
    resp.parent.mkdir(parents=True, exist_ok=True)
    tok_cfg = getattr(cfg.model, "tokenizer", None)
    max_length = int(getattr(tok_cfg, "model_max_length", 2048) or 2048)

    verdicts = []
    with open(resp, "w", encoding="utf-8") as rf:
        for i, row in enumerate(rows):
            prompt = _build_prompt(tokenizer, row, use_chat_template)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            text = _strip_thinking(text)
            rf.write(json.dumps({"id": row.get("id", i), "result": text}, ensure_ascii=False) + "\n")
            rf.flush()
            verdicts.append(evaluate_bfcl_sample(category, row, text))
            if (i + 1) % 25 == 0:
                print(f"[bfcl] {category}: {i + 1}/{len(rows)} done", flush=True)

    metrics = summarize_bfcl_results(category, verdicts, len(rows))
    metrics["model"] = str(cfg.model.pretrained_model_name_or_path)
    if ap:
        metrics["adapter_path"] = str(ap)
    outp = Path(str(cfg.output_json))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
