"""MMStar Benchmark Evaluation: multimodal multiple-choice with local Qwen3.5 model.

MMStar (Are We on the Right Way for Evaluating Large Vision-Language Models?)
contains 1500 vision-indispensable multiple-choice samples; each sample has one
image and a question with embedded "Options: A: ... B: ... C: ... D: ...".

Supports optional online TTA (TENT / TLM): per sample the model first generates
an answer, then performs one adaptation step on the same unlabeled sample.
"""

from __future__ import annotations

import os

# 需在首次 CUDA 分配前生效；允许用户在外部已设置时覆盖。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from openttl.eval.answer_extraction import (  # noqa: E402
    extract_answer_letter,
    slugify_model_for_csv,
    strip_thinking_block,
)

_SYSTEM_PROMPT = (
    "You solve multiple-choice questions about images. "
    "Be precise and base your answer only on the provided image and question. "
    "Respond with exactly one chosen letter A, B, C, or D. "
    "After any internal reasoning, your final visible line must be exactly: Answer: X "
    "(X is A, B, C, or D)."
)


def format_mmstar_prompt(question: str) -> str:
    """MMStar 题干已内嵌 'Options: A: ... B: ...'，仅需追加作答指令。"""
    return f"{question}\n\nAnswer with the letter only. Final Answer:"


def _resolve_mmstar_csv_path(
    output_json: str,
    model_cfg_name: Optional[str],
    model_id: str,
) -> str:
    out_dir = Path(output_json).parent
    raw = (model_cfg_name or "").strip() or model_id
    model_slug = slugify_model_for_csv(raw)
    model_slug = re.sub(r"[^\w\-]+", "_", model_slug).strip("_") or "model"
    return str(out_dir / f"mmstar_{model_slug}.csv")


def _mmstar_tta_step(
    runner: Any,
    adapter: Any,
    cfg: DictConfig,
    *,
    chat_prompt_text: str,
    prompt_plain: str,
    use_images_in_batch: Optional[List[Any]],
    messages: Optional[List[Any]],
    response: str,
    max_tok: int,
    device: torch.device,
) -> None:
    """Single online TTA step; does not accept gold labels (signature-enforced isolation).

    ``messages`` must embed the actual PIL image (``{"type": "image", "image": pil}``)
    so the adapter re-applies the chat template with the real image and the number of
    ``<|image_pad|>`` tokens matches the vision encoder output exactly.
    """
    from openttl.online.batching import (
        build_tta_batch,
        strategy_suppresses_response,
        strategy_to_label_mode,
    )

    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "tent").lower()
    lm = strategy_to_label_mode(strat_name, prompt_only_tta=False)
    eff_response = None if strategy_suppresses_response(strat_name) else response
    _enable_think = bool(OmegaConf.select(cfg, "mmstar.enable_thinking", default=True))
    batch = build_tta_batch(
        adapter,
        chat_prompt_text=chat_prompt_text,
        prompt_plain=prompt_plain,
        images=use_images_in_batch,
        messages=messages,
        response=eff_response,
        max_length=max_tok,
        device=device,
        label_mode=lm,
        enable_thinking=_enable_think,
        mm_encode_like_inference=True,
    )
    runner.update(batch)


def evaluate_mmstar(
    adapter: Any,
    inference: Any,
    runner: Optional[Any],
    cfg: DictConfig,
    max_examples: Optional[int] = None,
    start_example: int = 0,
    device: torch.device = torch.device("cuda"),
    csv_path: Optional[str] = None,
) -> tuple[dict, list]:
    """Run MMStar evaluation (HF/SGLang engine; optional online TTA)."""
    from openttl.data.mmstar import load_mmstar_dataset

    max_tok = int(OmegaConf.select(cfg, "online.max_length") or 4096)
    gen_max = int(OmegaConf.select(cfg, "mmstar.max_new_tokens") or 16000)
    local_root = OmegaConf.select(cfg, "mmstar_local_root")
    hf_path = str(OmegaConf.select(cfg, "mmstar_hf_path") or "Lin-Chen/MMStar")
    mmstar_enable_thinking = bool(
        OmegaConf.select(cfg, "mmstar.enable_thinking", default=True)
    )
    # MMA（Modality Mirror Alignment）：冻结参数的中间层偏移对齐，独立于 online TTA
    mma_enabled = bool(OmegaConf.select(cfg, "mma.enabled") or False)

    correct = 0
    total = 0
    unextracted = 0
    results = []
    # 按 category / l2_category 统计 {name: [correct, total]}
    per_category: Dict[str, List[int]] = {}
    per_l2: Dict[str, List[int]] = {}

    csv_file = None
    csv_writer: Optional[Any] = None
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(p, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["predict", "target", "score"])

    print(f"Loading MMStar dataset (local_root={local_root}, hf_path={hf_path})")
    if start_example > 0:
        print(f"Skipping first {start_example} examples")
    if max_examples:
        print(f"Evaluating on {max_examples} examples")

    for idx, row in enumerate(
        load_mmstar_dataset(
            local_root=local_root,
            hf_path=hf_path,
            max_samples=max_examples,
            start_example=start_example,
        )
    ):
        question = str(row["question"])
        image = row["image"]
        category = str(row.get("category") or "unknown")
        l2_category = str(row.get("l2_category") or "unknown")
        global_idx = idx + start_example

        prompt = format_mmstar_prompt(question)

        response = ""
        pred_letter: Optional[str] = None
        try:
            if adapter.supports_vision:
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                chat_prompt_text = adapter.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=mmstar_enable_thinking,
                )
                img_arg: Any = [image]
                use_images_in_batch: Optional[List[Any]] = [image]
                # TTA 时必须把真实 PIL 图像嵌入 messages，让 apply_chat_template
                # 根据实际图像尺寸计算正确的 <|image_pad|> 数量。
                messages_for_tta: Optional[List[Any]] = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            else:
                chat_prompt_text = prompt
                img_arg = None
                use_images_in_batch = None
                messages_for_tta = None

            sampling = {
                "max_new_tokens": gen_max,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            if mma_enabled:
                # MMA：前向至 l* 层提取视觉/文本隐状态，优化偏移 δ 后注入并生成
                from openttl.strategies.mma import mma_generate

                raw_out = mma_generate(
                    model=inference.model,
                    adapter=adapter,
                    prompt_text=chat_prompt_text,
                    images=img_arg,
                    device=device,
                    l_star=OmegaConf.select(cfg, "mma.l_star"),
                    K=int(OmegaConf.select(cfg, "mma.K") or 5),
                    eta=float(OmegaConf.select(cfg, "mma.eta") or 0.1),
                    lambda_reg=float(OmegaConf.select(cfg, "mma.lambda_reg") or 0.01),
                    max_new_tokens=gen_max,
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                )
            else:
                raw_out = inference.generate(
                    chat_prompt_text,
                    image_data=img_arg,
                    sampling_params=sampling,
                    lora_name=inference.current_lora_name,
                )
            response = str(raw_out).strip()

            if response and not any(l in response for l in ['A', 'B', 'C', 'D']):
                after_think = strip_thinking_block(response)
                after_think = re.sub(r'<\|[^|]+\|>', '', after_think).strip()
                if after_think:
                    response = after_think

            pred_letter = extract_answer_letter(response)

            if pred_letter is None:
                unextracted += 1
                print(
                    f"  Warning: failed to extract answer for example {global_idx}; "
                    f"response head: {repr(response[:300])}"
                )

            if runner is not None and runner.enabled() and not response.startswith("ERROR"):
                try:
                    _mmstar_tta_step(
                        runner,
                        adapter,
                        cfg,
                        chat_prompt_text=chat_prompt_text,
                        prompt_plain=prompt,
                        use_images_in_batch=use_images_in_batch,
                        messages=messages_for_tta,
                        response=response,
                        max_tok=max_tok,
                        device=device,
                    )
                except Exception as te:
                    print(f"  Warning: TTA update failed at example {global_idx}: {te}")

        except Exception as e:
            print(f"Error processing example {global_idx}: {e}")
            pred_letter = None
            response = f"ERROR: {str(e)}"

        gold_answer = str(row["answer"]).strip().upper()
        is_correct = pred_letter == gold_answer if pred_letter else False
        if is_correct:
            correct += 1
        total += 1
        for table, key in ((per_category, category), (per_l2, l2_category)):
            slot = table.setdefault(key, [0, 0])
            slot[0] += int(is_correct)
            slot[1] += 1

        results.append({
            'idx': row.get('index', global_idx),
            'question': question[:100] + '...' if len(question) > 100 else question,
            'answer': gold_answer,
            'prediction': pred_letter,
            'correct': is_correct,
            'category': category,
            'l2_category': l2_category,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            acc = correct / total if total > 0 else 0.0
            print(
                f"Example {global_idx + 1}: Predicted={pred_letter}, Gold={gold_answer}, "
                f"Correct={is_correct} | Running Accuracy: {acc:.3f}"
            )

        if csv_writer is not None:
            csv_writer.writerow(
                [pred_letter or "", gold_answer, "true" if is_correct else "false"]
            )
            csv_file.flush()

        # 释放本样本图像与 TTA 张量引用，便于 CUDA 分配器回收。
        del image, img_arg, use_images_in_batch, messages_for_tta

    if csv_file is not None:
        csv_file.close()

    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'unextracted': unextracted,
        'max_examples': max_examples,
        'start_example': start_example,
        'per_category': {
            k: {'correct': c, 'total': t, 'accuracy': (c / t if t else 0.0)}
            for k, (c, t) in sorted(per_category.items())
        },
        'per_l2_category': {
            k: {'correct': c, 'total': t, 'accuracy': (c / t if t else 0.0)}
            for k, (c, t) in sorted(per_l2.items())
        },
    }

    return metrics, results


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_mmstar")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point."""
    from openttl.adapters.registry import extract_model_cfg
    from openttl.models.loader import load_adapter
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.tta_runner import OnlineTTARunner

    max_examples = OmegaConf.select(cfg, "max_examples")
    start_example = int(OmegaConf.select(cfg, "start_example", default=0))
    output_json = str(cfg.output_json)

    model_id = str(
        OmegaConf.select(cfg, "model.pretrained_model_name_or_path")
        or OmegaConf.select(cfg, "model_path")
        or ""
    )
    if not model_id:
        raise ValueError("请在配置中设置 model.pretrained_model_name_or_path（或兼容字段 model_path）")
    print(f"Loading model adapter / processor from: {model_id}")

    try:
        from hydra.core.hydra_config import HydraConfig

        model_cfg_name = HydraConfig.get().runtime.choices.get("model")
    except Exception:
        model_cfg_name = None
    output_csv = _resolve_mmstar_csv_path(output_json, model_cfg_name, model_id)

    mc = extract_model_cfg(cfg)
    adapter = load_adapter(cfg)
    adapter.load_processor(mc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = str(OmegaConf.select(cfg, "inference.backend") or "hf").lower()
    online_on = bool(OmegaConf.select(cfg, "online.enabled") or False)
    peft_on = bool(OmegaConf.select(cfg, "model.peft.enabled") or False)

    train_model = None
    runner = None
    initial_lora_path = None

    if online_on:
        if not peft_on:
            raise ValueError("online.enabled=true 需要 model.peft.enabled=true")
        train_model = adapter.load_model(mc)
        train_model = inject_lora(train_model, cfg.model.peft)
        train_model.to(device)

    if backend == "sglang":
        from openttl.inference.sglang_engine import build_sglang_engine_from_omegaconf

        if online_on:
            # 仅 SGLang 需要在启动引擎前导出 tta_v0 权重；HF 后端原位更新，无需导出。
            inference_cfg = OmegaConf.select(cfg, "inference")
            initial_lora_path = str(
                OnlineTTARunner.initial_adapter_path(cfg, train_model, inference_cfg)
            )
        tok = adapter.tokenizer()
        infer = build_sglang_engine_from_omegaconf(
            cfg.model,
            cfg.inference,
            tok,
            initial_lora_path if (online_on and peft_on) else None,
        )
    elif backend == "hf":
        from openttl.inference.hf_engine import HuggingFaceEngine

        if train_model is None:
            train_model = adapter.load_model(mc)
            if peft_on:
                train_model = inject_lora(train_model, cfg.model.peft)
            train_model.to(device)
        infer = HuggingFaceEngine(train_model, adapter, device)
    else:
        raise ValueError(f"未知 inference.backend: {backend}")

    if online_on:
        runner = OnlineTTARunner(
            cfg,
            model=train_model,
            adapter=adapter,
            inference=infer,
            device=device,
        )

    print("Model backend ready!")
    print("Evaluating on MMStar benchmark...")

    metrics, results = evaluate_mmstar(
        adapter=adapter,
        inference=infer,
        runner=runner,
        cfg=cfg,
        max_examples=max_examples,
        start_example=start_example,
        device=device,
        csv_path=output_csv,
    )

    try:
        shutdown = getattr(infer, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("MMStar Evaluation Results")
    print("=" * 50)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metrics': metrics,
        'results': results,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }

    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {output_path}")
    print(f"Per-example CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
