"""ERQA Benchmark Evaluation: Multimodal question answering with local Qwen3.5 model.

This script evaluates a vision-language model on the ERQA benchmark, which consists
of multimodal interleaved images and text formatted as multiple-choice questions.
"""

from __future__ import annotations

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
    # Build the prompt with explicit instruction to output a single letter.
    # Use few-shot-style formatting to constrain the model's output format.
    instruction = (
        "\n\nYou must respond with ONLY a single letter: A, B, C, or D."
        " Do NOT explain or reason. Output format example: B"
    )

    if len(images) == 0:
        prompt = question + instruction
        return prompt, []
    elif len(images) == 1:
        prompt = f"Answer the following question based on the image.\n\n{question}{instruction}"
        return prompt, images
    else:
        image_refs = "\n".join([f"Image {i+1}:" for i in range(len(images))])
        prompt = f"Answer the following question based on the provided images.\n\n{image_refs}\n\n{question}{instruction}"
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

    # Try structured patterns first on the answer part
    answer_patterns = [
        r'[Aa]nswer' + r'\s*[:\uff1a]' + r'\s*([A-D])',
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


def evaluate_erqa(
    model: torch.nn.Module,
    processor: Any,
    dataset_path: str,
    max_examples: Optional[int] = None,
    start_example: int = 0,
    device: torch.device = torch.device("cuda"),
    is_multimodal: bool = True,
) -> dict:
    """Run ERQA evaluation.

    Args:
        model: The vision-language model
        processor: The processor/tokenizer for the model
        dataset_path: Path to ERQA TFRecord file
        max_examples: Maximum number of examples to evaluate
        start_example: Index of the first example to evaluate (skip before this)
        device: Device to run inference on

    Returns:
        Dictionary with evaluation metrics
    """
    from openttl.data.erqa import load_erqa_dataset

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    results = []

    print(f"Loading ERQA dataset from: {dataset_path}")
    if start_example > 0:
        print(f"Skipping first {start_example} examples, starting from example {start_example}")
    if max_examples:
        print(f"Evaluating on {max_examples} examples")

    for idx, example in enumerate(load_erqa_dataset(dataset_path, max_examples, start_example)):
        question = example['question']
        images = example['images']
        answer = example['answer']
        question_type = example['question_type']
        visual_indices = example['visual_indices']

        # Format prompt
        prompt, image_list = format_erqa_prompt(question, images, visual_indices)

        # Prepare input for model
        try:
            if is_multimodal and len(image_list) > 0:
                # Multimodal input (vision-language model)
                # Each image needs its own {"type": "image"} entry in the content list.
                image_entries = [{"type": "image"} for _ in image_list]
                messages = [
                    {
                        "role": "system",
                        "content": "You are a multiple-choice question answering assistant. You MUST respond with ONLY a single letter (A, B, C, or D). Never explain your reasoning.",
                    },
                    {
                        "role": "user",
                        "content": image_entries + [{"type": "text", "text": prompt}]
                    },
                ]

                # Apply chat template to insert image tokens
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                # Process with text and images
                inputs = processor(
                    text=[text],
                    images=image_list,
                    return_tensors="pt"
                )
            else:
                # Text-only input (language-only model or no images)
                if not is_multimodal and len(images) > 0:
                    print(f"  Warning: Skipping {len(images)} image(s) - model is text-only")

                inputs = processor(
                    text=prompt,
                    return_tensors="pt"
                )

            # Move inputs to device
            input_ids = inputs['input_ids'].to(device)

            # Prepare model inputs - include all required keys for Qwen3.5
            model_inputs = {
                'input_ids': input_ids,
            }

            # Add attention_mask if present
            if 'attention_mask' in inputs:
                model_inputs['attention_mask'] = inputs['attention_mask'].to(device)

            # Add pixel_values if present (for vision models)
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                model_inputs['pixel_values'] = inputs['pixel_values'].to(device)

            # Add image_grid_thw if present (required for Qwen3.5)
            if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
                model_inputs['image_grid_thw'] = inputs['image_grid_thw'].to(device)

            # Add mm_token_type_ids if present (required for Qwen3.5)
            if 'mm_token_type_ids' in inputs and inputs['mm_token_type_ids'] is not None:
                model_inputs['mm_token_type_ids'] = inputs['mm_token_type_ids'].to(device)

            # Generate response - generous max_new_tokens because Qwen3.5
            # thinking mode can consume many tokens before producing the answer.
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    output_scores=False,
                    return_dict_in_generate=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            # Debug: check outputs type and content
            if outputs is None:
                raise ValueError("model.generate() returned None")

            # Handle different output formats
            if hasattr(outputs, 'sequences'):
                output_sequences = outputs.sequences
            elif isinstance(outputs, (list, tuple)):
                output_sequences = outputs[0] if len(outputs) > 0 else None
                if output_sequences is None:
                    raise ValueError("model.generate() returned empty tuple")
            else:
                output_sequences = outputs

            # Decode response
            generated_tokens = output_sequences[0][input_ids.shape[1]:]
            num_generated = generated_tokens.numel()

            if num_generated == 0:
                response = ""
                response_clean = ""
                print(f"  Warning: Empty generation for example {idx}")
            else:
                # PRIMARY: decode with skip_special_tokens=True for clean text.
                response_clean = processor.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()
                response = response_clean

                # FALLBACK: if the clean response contains no A-D letter at all,
                # try decoding with special tokens preserved to detect the
                # thinking-block boundary, then extract text after it.
                if response and not any(l in response for l in ['A', 'B', 'C', 'D']):
                    response_raw = processor.decode(
                        generated_tokens,
                        skip_special_tokens=False
                    ).strip()
                    after_think = _strip_thinking_block(response_raw)
                    # Remove any remaining special tokens like <|im_end|>
                    after_think = re.sub(r'<\|[^|]+\|>', '', after_think).strip()
                    if after_think:
                        response = after_think

            # Extract answer
            pred_letter = extract_answer_letter(response)

            # Detailed debug output when extraction fails
            if pred_letter is None:
                print(f"\n  === DEBUG: Failed to extract answer for example {idx} ===")
                print(f"  Generated tokens: {num_generated}")
                print(f"  Clean decode (skip_special=True): {repr(response_clean[:500])}")
                if num_generated > 0:
                    response_raw = processor.decode(
                        generated_tokens,
                        skip_special_tokens=False
                    ).strip()
                    print(f"  Raw decode  (skip_special=False): {repr(response_raw[:500])}")
                    has_think_close = '\u003c/think\u003e' in response_raw
                    print(f"  Contains </think> tag: {has_think_close}")
                    if has_think_close:
                        after = response_raw.split('\u003c/think\u003e')[-1].strip()
                        print(f"  Text after </think>: {repr(after[:300])}")
                print(f"  Current response used for extraction: {repr(response[:500])}")
                # Show which patterns were tested
                answer_part = _strip_thinking_block(response)
                print(f"  Answer part after strip_thinking: {repr(answer_part[:300])}")
                print(f"  =============================={'='*40}\n")

            # ---- Explicit GPU memory cleanup ----
            del inputs, model_inputs, outputs, output_sequences, generated_tokens
            if hasattr(model, 'past_key_values'):
                model.past_key_values = None

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            pred_letter = None
            response = f"ERROR: {str(e)}"

        # Check if correct
        is_correct = pred_letter == answer.upper() if pred_letter else False
        if is_correct:
            correct += 1
        total += 1

        # Store result
        results.append({
            'idx': idx,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'answer': answer,
            'prediction': pred_letter,
            'correct': is_correct,
            'num_images': len(images),
            'question_type': question_type,
        })

        # Print progress
        if (idx + 1) % 10 == 0 or idx == 0:
            acc = correct / total if total > 0 else 0.0
            print(f"Example {idx+1}: Predicted={pred_letter}, Gold={answer}, Correct={is_correct} | Running Accuracy: {acc:.3f}")

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
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    # Get configuration
    tfrecord_path = str(cfg.tfrecord_path)
    model_path = str(cfg.model_path)
    max_examples = OmegaConf.select(cfg, "max_examples")
    start_example = int(OmegaConf.select(cfg, "start_example", default=0))
    output_json = str(cfg.output_json)

    print(f"Loading model from: {model_path}")

    # Try to load as vision-language model first, fall back to text-only
    try:
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = model_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        is_multimodal = True
        print("Loaded as multimodal (vision-language) model")
    except Exception as e:
        print(f"Not a vision-language model ({e}), loading as text-only model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        is_multimodal = False
        processor = tokenizer
        print("Loaded as text-only model (images will be skipped)")

    print("Model loaded successfully!")
    print(f"Evaluating on ERQA benchmark...")

    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, results = evaluate_erqa(
        model=model,
        processor=processor,
        dataset_path=tfrecord_path,
        max_examples=max_examples,
        start_example=start_example,
        device=device,
        is_multimodal=is_multimodal,
    )

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


if __name__ == "__main__":
    main()
