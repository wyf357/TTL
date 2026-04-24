"""ERQA Benchmark Evaluation: Multimodal question answering with local Qwen3.5 model.

This script evaluates a vision-language model on the ERQA benchmark, which consists
of multimodal interleaved images and text formatted as multiple-choice questions.
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
from typing import Any, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Qwen3.5 thinking-mode tags (constructed via unicode escapes to avoid
# XML-parsing issues in tool calls)
_THINK_CLOSE = "\u003c/think\u003e"


def _strip_thinking_block(text: str) -> str:
    """Remove Qwen3.5 thinking block from response text.

    Qwen3.5 generates a thinking block before the actual answer.
    We detect the closing tag and return only the text that follows it.
    If no thinking block is found, the original text is returned unchanged.
    """
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE)[-1].strip()
    return text


def _slugify_model_for_csv(name: str) -> str:
    """Normalize Hydra model group name or checkpoint dirname for CSV basename.

    Examples: ``qwen35_2b`` -> ``qwen35_2B``; ``Qwen3.5-2B`` path segment -> ``qwen3_5_2B``.
    """
    s = name.strip()
    if "/" in s or "\\" in s:
        s = Path(s).name
    s = s.lower().replace("-", "_")
    m = re.match(r"^(.+_)(\d+)(b)$", s)
    if m:
        return m.group(1) + m.group(2) + "B"
    return s


def _resolve_erqa_csv_path(
    output_json: str,
    tfrecord_path: str,
    model_cfg_name: Optional[str],
    model_id: str,
) -> str:
    """``{dataset_stem}_{model_slug}.csv`` next to ``output_json`` (e.g. ``erqa_qwen35_2B.csv``)."""
    out_dir = Path(output_json).parent
    dataset_stem = Path(tfrecord_path).stem or "erqa"
    raw = (model_cfg_name or "").strip() or model_id
    model_slug = _slugify_model_for_csv(raw)
    model_slug = re.sub(r"[^\w\-]+", "_", model_slug).strip("_") or "model"
    return str(out_dir / f"{dataset_stem}_{model_slug}.csv")


def format_erqa_prompt(
    question: str,
    images: List[Image.Image],
    visual_indices: Any,
) -> tuple[str, List[Image.Image]]:
    """Format ERQA question with interleaved images for multimodal model.

    The visual_indices indicate where images should be placed in the question text.
    We construct a prompt that properly interleaves images and text.

    Args:
        question: The question text
        images: List of PIL Image objects
        visual_indices: Indices indicating image placement positions

    Returns:
        Tuple of (formatted prompt text, ordered list of images)
    """
    instruction = "\n\nChoose one letter A/B/C/D based only on the image. Final Answer:"

    if len(images) == 0:
        prompt = question + instruction
        return prompt, []
    elif len(images) == 1:
        prompt = (
            "Answer the multiple-choice question below using the single image.\n\n"
            f"{question}{instruction}"
        )
        return prompt, images
    else:
        image_refs = "\n".join([f"Image {i+1}:" for i in range(len(images))])
        prompt = (
            "Answer the multiple-choice question below using all images when relevant.\n\n"
            f"{image_refs}\n\n{question}{instruction}"
        )
        return prompt, images


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter (A, B, C, or D) from model response.

    Handles Qwen3.5 thinking mode by first stripping the thinking block,
    then searching for the answer letter in the remaining text.

    Args:
        response: Raw model response text

    Returns:
        Extracted answer letter or None if not found
    """
    # First strip any thinking block
    answer_part = _strip_thinking_block(response)

    # Prefer the last "Answer: X" in the visible part: few-shot prompts may list several
    # "Answer: …" lines; the instruction requires the final line to be the true choice.
    _ans_marks = re.findall(r"[Aa]nswer\s*[:\uff1a]\s*([A-D])", answer_part)
    if _ans_marks:
        return _ans_marks[-1].upper()

    # Try structured patterns first on the answer part
    answer_patterns = [
        r'[Tt]he' + r'\s+answer' + r'\s+is' + r'\s+([A-D])',
        r'[Oo]ption' + r'\s+([A-D])' + r'\b',
        r'[Cc]hoice' + r'\s+([A-D])' + r'\b',
        r'^' + r'\s*([A-D])' + r'\s*$',
        r'^' + r'\s*([A-D])' + r'[\.\uff0c,\s]',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, answer_part, re.MULTILINE)
        if match:
            return match.group(1).upper()

    # If no pattern matches in answer part, try the first 100 chars
    first_part = answer_part[:100]
    for letter in ['A', 'B', 'C', 'D']:
        if letter in first_part:
            return letter

    # Any "Answer: X" in the full raw response (thinking + visible); use last match.
    _ans_all = re.findall(r"[Aa]nswer\s*[:\uff1a]\s*([A-D])", response)
    if _ans_all:
        return _ans_all[-1].upper()

    # Last resort: scan the ENTIRE response (including thinking)
    # Add patterns that commonly appear inside the thinking block itself,
    # where the model states its conclusion before formatting the answer.
    thinking_patterns = answer_patterns + [
        r'[Tt]herefore[\s,]+(?:the answer is\s+)?([A-D])',
        r'[Ss]o(?:\s+the answer is)?\s+([A-D])',
        r'[Aa]ccordingly[\s,]+(?:the answer is\s+)?([A-D])',
        r'[Tt]hus[\s,]+(?:the answer is\s+)?([A-D])',
        r'[Ii] conclude(?:\s+that)?(?:\s+the answer is)?\s+([A-D])',
    ]
    for pattern in thinking_patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return match.group(1).upper()

    return None


_VISION_FALLBACK_WARNED = False


def _erqa_tta_step(
    runner: Any,
    adapter: Any,
    cfg: DictConfig,
    *,
    chat_prompt_text: str,
    prompt_plain: str,
    use_images_in_batch: Optional[List[Any]],
    messages: Optional[List[Any]] = None,
    response: str,
    max_tok: int,
    device: torch.device,
) -> None:
    """Single online TTA step; does not accept gold labels (signature-enforced isolation).

    ``messages`` should carry the full conversation structure with actual PIL images
    embedded (``{"type": "image", "image": pil_img}``).  When provided, the adapter's
    ``build_forward_inputs`` will re-apply the chat template with the real images so that
    the number of ``<|image_pad|>`` tokens matches the vision encoder output exactly,
    preventing "Image features and image tokens do not match" errors.
    """
    from openttl.online.batching import (
        build_tta_batch,
        strategy_suppresses_response,
        strategy_to_label_mode,
    )

    strat_name = str(OmegaConf.select(cfg, "strategy.name") or "tent").lower()
    lm = strategy_to_label_mode(strat_name, prompt_only_tta=False)
    eff_response = None if strategy_suppresses_response(strat_name) else response
    _enable_think = bool(
        OmegaConf.select(cfg, "erqa.enable_thinking", default=True)
    )
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
        # 与 SGLang generate 使用同一 ``chat_prompt_text``（含 add_generation_prompt=True）
        mm_encode_like_inference=True,
    )
    runner.update(batch)


def evaluate_erqa(
    adapter: Any,
    inference: Any,
    runner: Optional[Any],
    dataset_path: str,
    cfg: DictConfig,
    max_examples: Optional[int] = None,
    start_example: int = 0,
    device: torch.device = torch.device("cuda"),
    csv_path: Optional[str] = None,
) -> dict:
    """Run ERQA evaluation（推理：SGLang / HF Engine；可选 online TTA）。"""
    from openttl.data.erqa import load_erqa_dataset
    from openttl.adapters.base import ModelAdapter
    from openttl.eval.erqa_gold_guard import DelayedGoldExample

    assert isinstance(adapter, ModelAdapter)
    max_tok = int(OmegaConf.select(cfg, "online.max_length") or 4096)
    gen_max = int(OmegaConf.select(cfg, "erqa.max_new_tokens") or 16000)

    correct = 0
    total = 0
    results = []

    csv_file = None
    csv_writer: Optional[Any] = None
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(p, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["predict", "target", "score"])

    print(f"Loading ERQA dataset from: {dataset_path}")
    if start_example > 0:
        print(f"Skipping first {start_example} examples, starting from example {start_example}")
    if max_examples:
        print(f"Evaluating on {max_examples} examples")

    global _VISION_FALLBACK_WARNED

    strict_gold = bool(OmegaConf.select(cfg, "erqa.strict_no_label_feedback") or False)
    erqa_enable_thinking = bool(
        OmegaConf.select(cfg, "erqa.enable_thinking", default=True)
    )

    for idx, example in enumerate(load_erqa_dataset(dataset_path, max_examples, start_example)):
        ex: Any = (
            DelayedGoldExample(dict(example), strict=True) if strict_gold else example
        )

        question = ex["question"]
        images = ex["images"]
        question_type = ex["question_type"]
        visual_indices = ex["visual_indices"]

        # Format prompt
        prompt, image_list = format_erqa_prompt(question, images, visual_indices)

        # Prepare input for model
        response_clean = ""
        num_generated = -1
        try:
            chat_prompt_text = ""
            img_arg: Any = None
            use_images_in_batch: Optional[List[Any]] = None
            messages_for_tta: Optional[List[Any]] = None

            if len(image_list) > 0 and adapter.supports_vision:
                image_entries = [{"type": "image"} for _ in image_list]
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You solve multiple-choice questions about scenes, objects, and embodied reasoning. "
                            "Be precise and avoid speculation beyond what the question and image(s) support. "
                            "Respond with exactly one chosen letter A, B, C, or D. "
                            "After any internal reasoning, your final visible line must be exactly: Answer: X "
                            "(X is A, B, C, or D)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": image_entries + [{"type": "text", "text": prompt}],
                    },
                ]
                chat_prompt_text = adapter.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=erqa_enable_thinking,
                )
                img_arg = image_list
                use_images_in_batch = image_list
                # TTA 时必须把真实 PIL 图像嵌入 messages，让 apply_chat_template
                # 根据实际图像尺寸计算正确的 <|image_pad|> 数量，否则会出现
                # "Image features and image tokens do not match" 错误。
                messages_for_tta = [
                    {
                        "role": "system",
                        "content": messages[0]["content"],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img} for img in image_list
                        ] + [{"type": "text", "text": prompt}],
                    },
                ]
            elif len(image_list) > 0 and not adapter.supports_vision:
                if not _VISION_FALLBACK_WARNED:
                    print(
                        "  Warning: examples contain images but the loaded processor has no vision support; "
                        "running text-only on the prompt (consider a native multimodal checkpoint)."
                    )
                    _VISION_FALLBACK_WARNED = True
                chat_prompt_text = prompt
                img_arg = None
                use_images_in_batch = None
            else:
                chat_prompt_text = prompt
                img_arg = None
                use_images_in_batch = None

            sampling = {
                "max_new_tokens": gen_max,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            raw_out = inference.generate(
                chat_prompt_text,
                image_data=img_arg,
                sampling_params=sampling,
                lora_name=inference.current_lora_name,
            )
            response = str(raw_out).strip()
            response_clean = response

            if response and not any(l in response for l in ['A', 'B', 'C', 'D']):
                after_think = _strip_thinking_block(response)
                after_think = re.sub(r'<\|[^|]+\|>', '', after_think).strip()
                if after_think:
                    response = after_think

            # Extract answer
            pred_letter = extract_answer_letter(response)

            # Detailed debug output when extraction fails
            if pred_letter is None:
                print(f"\n  === DEBUG: Failed to extract answer for example {idx + start_example} ===")
                print(f"  Response length (chars): {len(response)}")
                print(f"  Clean text: {repr(response_clean[:500])}")
                print(f"  Current response used for extraction: {repr(response[:500])}")
                # Show which patterns were tested
                answer_part = _strip_thinking_block(response)
                print(f"  Answer part after strip_thinking: {repr(answer_part[:300])}")
                print(f"  =============================={'='*40}\n")

            if runner is not None and runner.enabled() and not str(response).startswith("ERROR"):
                try:
                    _erqa_tta_step(
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
                    print(f"  Warning: TTA update failed at example {idx + start_example}: {te}")

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            pred_letter = None
            response = f"ERROR: {str(e)}"

        if strict_gold:
            assert isinstance(ex, DelayedGoldExample)
            ex.allow_gold_read()

        gold_answer = str(ex["answer"]).strip()

        # Check if correct (gold only after adaptation step)
        is_correct = pred_letter == gold_answer.upper() if pred_letter else False
        if is_correct:
            correct += 1
        total += 1

        # Store result
        results.append({
            'idx': idx,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'answer': gold_answer,
            'prediction': pred_letter,
            'correct': is_correct,
            'num_images': len(images),
            'question_type': question_type,
        })

        # Print progress (use global index = idx + start_example + 1)
        global_idx = idx + start_example + 1
        if (idx + 1) % 10 == 0 or idx == 0:
            acc = correct / total if total > 0 else 0.0
            print(f"Example {global_idx}: Predicted={pred_letter}, Gold={gold_answer}, Correct={is_correct} | Running Accuracy: {acc:.3f}")

        if csv_writer is not None:
            target_str = str(gold_answer).strip().upper()
            pred_str = pred_letter if pred_letter is not None else ""
            csv_writer.writerow(
                [pred_str, target_str, "true" if is_correct else "false"]
            )
            csv_file.flush()

        # Release per-example image objects and TTA tensors so that PyTorch's CUDA
        # allocator can consolidate freed blocks before the next iteration.
        del images, image_list, use_images_in_batch, img_arg

    if csv_file is not None:
        csv_file.close()

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'dataset_path': dataset_path,
        'max_examples': max_examples,
    }

    return metrics, results


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_erqa")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point."""
    from openttl.adapters.registry import extract_model_cfg
    from openttl.models.loader import load_adapter
    from openttl.models.lora_wrapper import inject_lora
    from openttl.online.tta_runner import OnlineTTARunner

    tfrecord_path = str(cfg.tfrecord_path)
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
    output_csv = _resolve_erqa_csv_path(
        output_json, tfrecord_path, model_cfg_name, model_id
    )

    mc = extract_model_cfg(cfg)
    adapter = load_adapter(cfg)
    adapter.load_processor(mc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = str(OmegaConf.select(cfg, "inference.backend") or "sglang").lower()
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
        initial_lora_path = str(OnlineTTARunner.initial_adapter_path(cfg, train_model, cfg.inference))

    if backend == "sglang":
        from openttl.inference.sglang_engine import build_sglang_engine_from_omegaconf

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
    print("Evaluating on ERQA benchmark...")

    metrics, results = evaluate_erqa(
        adapter=adapter,
        inference=infer,
        runner=runner,
        dataset_path=tfrecord_path,
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

    # Print final results
    print("\n" + "="*50)
    print("ERQA Evaluation Results")
    print("="*50)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metrics': metrics,
        'results': results,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }

    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2)
    )
    print(f"\nResults saved to: {output_path}")
    print(f"Per-example CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
