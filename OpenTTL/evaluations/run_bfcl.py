"""BFCL (Berkeley Function Calling Leaderboard) 评测：函数调用预测 + AST 评分。

可选 online TTA：每样本先 generate/评分，再对 prompt 做无监督 LoRA 更新（与 run_mmlu 同序）。
数据与评分说明见 src/openttl/data/bfcl.py 与 src/openttl/eval/bfcl_eval.py。
产物：
  - output_json: 指标（类别、样本数、准确率等）
  - result_jsonl: 逐样本预测（{"id":..., "result":...}，兼容官方 answer 格式）
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def _generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
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
    return _strip_thinking(text)


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_bfcl")
def main(cfg: DictConfig) -> None:
    from peft import PeftModel

    from openttl.data.bfcl import load_bfcl_category
    from openttl.eval.bfcl_eval import evaluate_bfcl_sample, summarize_bfcl_results
    from openttl.models.loader import load_adapter, load_causal_lm, load_tokenizer
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.batching import build_tta_batch, strategy_to_label_mode

    category = str(cfg.category)
    max_s = OmegaConf.select(cfg, "max_samples")
    max_new_tokens = int(OmegaConf.select(cfg, "max_new_tokens") or 256)
    use_chat_template = bool(
        OmegaConf.select(cfg, "use_chat_template")
        if OmegaConf.select(cfg, "use_chat_template") is not None
        else True
    )
    online_on = bool(OmegaConf.select(cfg, "online.enabled") or False)
    peft_on = bool(OmegaConf.select(cfg, "model.peft.enabled") or False)
    ap = OmegaConf.select(cfg, "adapter_path")

    if online_on and bool(OmegaConf.select(cfg, "merge_lora") or False):
        raise ValueError("online.enabled 与 merge_lora 不能同时启用（merge 后无 PEFT 适配器可更新）")
    if online_on and not peft_on and not ap:
        raise ValueError("online.enabled=true 需要 PEFT（model.peft.enabled 或 adapter_path）")

    adapter = None
    runner = None
    tta_losses: List[float] = []

    if online_on:
        adapter = load_adapter(cfg)
        adapter.load_processor(cfg.model)
        tokenizer = adapter.tokenizer()
        base = adapter.load_model(cfg.model)
        if ap:
            train_model = PeftModel.from_pretrained(base, ap)
        else:
            train_model = inject_lora(base, cfg.model.peft)
        if bool(OmegaConf.select(cfg, "merge_lora") or False) and hasattr(
            train_model, "merge_and_unload"
        ):
            train_model = train_model.merge_and_unload()

        if getattr(train_model, "hf_device_map", None) is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_model.to(dev)
        else:
            dev = next(train_model.parameters()).device

        from openttl.inference.hf_engine import HuggingFaceEngine
        from openttl.online.tta_runner import OnlineTTARunner

        infer = HuggingFaceEngine(train_model, adapter, dev)
        runner = OnlineTTARunner(
            cfg,
            model=train_model,
            adapter=adapter,
            inference=infer,
            device=dev,
        )
        model = train_model
    else:
        tokenizer = load_tokenizer(cfg.model)
        model = load_causal_lm(cfg.model)
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
    gen_max_length = int(getattr(tok_cfg, "model_max_length", 2048) or 2048)
    tta_max_length = int(
        OmegaConf.select(cfg, "online.max_length") or gen_max_length
    )
    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "tent").lower()

    verdicts = []
    with open(resp, "w", encoding="utf-8") as rf:
        for i, row in enumerate(rows):
            prompt = _build_prompt(tokenizer, row, use_chat_template)
            text = _generate_one(
                model,
                tokenizer,
                prompt,
                device=dev,
                max_length=gen_max_length,
                max_new_tokens=max_new_tokens,
            )
            rf.write(
                json.dumps({"id": row.get("id", i), "result": text}, ensure_ascii=False)
                + "\n"
            )
            rf.flush()
            verdicts.append(evaluate_bfcl_sample(category, row, text))

            if runner is not None and runner.enabled():
                lm = strategy_to_label_mode(strat_name, prompt_only_tta=True)
                batch = build_tta_batch(
                    adapter,
                    chat_prompt_text=prompt,
                    prompt_plain=prompt,
                    images=None,
                    response=None,
                    max_length=tta_max_length,
                    device=dev,
                    label_mode=lm,
                    enable_thinking=False,
                )
                try:
                    loss_val = runner.update(batch)
                    tta_losses.append(loss_val)
                except Exception as e:
                    print(f"Warning: TTA update failed on sample {i}: {e}", flush=True)

            if (i + 1) % 25 == 0:
                extra = ""
                if tta_losses:
                    extra = (
                        f" tta_loss_mean={sum(tta_losses)/len(tta_losses):.4f}"
                        f" last={tta_losses[-1]:.4f}"
                    )
                print(f"[bfcl] {category}: {i + 1}/{len(rows)} done{extra}", flush=True)

    metrics = summarize_bfcl_results(category, verdicts, len(rows))
    metrics["model"] = str(cfg.model.pretrained_model_name_or_path)
    metrics["online.enabled"] = online_on
    metrics["strategy"] = strat_name if online_on else None
    if ap:
        metrics["adapter_path"] = str(ap)
    if tta_losses:
        finite = [x for x in tta_losses if x == x]  # drop NaN
        metrics["tta_updates"] = len(tta_losses)
        metrics["tta_updates_finite"] = len(finite)
        if finite:
            metrics["tta_loss_mean"] = sum(finite) / len(finite)
        else:
            metrics["tta_loss_mean"] = None
    outp = Path(str(cfg.output_json))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
