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
    # For Qwen3.5-VL, we use a simple format:
    # Present all images first, then the question
    # This works well with the model's multimodal chat template
    
    # Build the prompt
    if len(images) == 0:
        # No images, just text
        prompt = question
        return prompt, []
    elif len(images) == 1:
        # Single image
        prompt = f"Please answer the following question based on the image:\n\n{question}"
        return prompt, images
    else:
        # Multiple images
        image_refs = "\n".join([f"Image {i+1}:" for i in range(len(images))])
        prompt = f"Please answer the following question based on the provided images:\n\n{image_refs}\n\n{question}"
        return prompt, images


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter (A, B, C, or D) from model response.
    
    Args:
        response: Raw model response text
        
    Returns:
        Extracted answer letter or None if not found
    """
    # Try to find single letter answer
    # Common patterns: "Answer: A", "A", "The answer is A", etc.
    
    # First, try to find standalone A, B, C, D
    # Look for patterns like "Answer: X" or "答案是X"
    patterns = [
        r'[Aa]nswer\s*[:：]\s*([A-D])',
        r'[Tt]he\s+answer\s+is\s+([A-D])',
        r'^\s*([A-D])\s*$',  # Just a single letter
        r'^\s*([A-D])[\.，,\s]',  # Letter followed by punctuation
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # If no pattern matches, try to find any A-D letter in the first 100 chars
    first_part = response[:100]
    for letter in ['A', 'B', 'C', 'D']:
        if letter in first_part:
            return letter
    
    return None


def evaluate_erqa(
    model: torch.nn.Module,
    processor: Any,
    dataset_path: str,
    max_examples: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
    is_multimodal: bool = True,
) -> dict:
    """Run ERQA evaluation.
    
    Args:
        model: The vision-language model
        processor: The processor/tokenizer for the model
        dataset_path: Path to ERQA TFRecord file
        max_examples: Maximum number of examples to evaluate
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
    if max_examples:
        print(f"Evaluating on {max_examples} examples")
    
    for idx, example in enumerate(load_erqa_dataset(dataset_path, max_examples)):
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
                # Qwen3.5 requires chat template format for multimodal input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Apply chat template to insert image tokens
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
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
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    output_scores=False,
                    return_dict_in_generate=False,  # Ensure we get tensor output
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            
            # Debug: check outputs type and content
            if outputs is None:
                raise ValueError("model.generate() returned None")
            
            # Handle different output formats
            if hasattr(outputs, 'sequences'):
                # transformers GenerateOutput object
                output_sequences = outputs.sequences
            elif isinstance(outputs, (list, tuple)):
                # Tuple output (some older versions)
                output_sequences = outputs[0] if len(outputs) > 0 else None
                if output_sequences is None:
                    raise ValueError("model.generate() returned empty tuple")
            else:
                # Direct tensor output (most common)
                output_sequences = outputs
            
            # Decode response
            # output_sequences should be a tensor, get the generated tokens (excluding input)
            generated_tokens = output_sequences[0][input_ids.shape[1]:]
            
            # Check if generated_tokens is valid
            if generated_tokens.numel() == 0:
                response = ""  # Empty generation
                print(f"  Warning: Empty generation for example {idx}")
            else:
                response = processor.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()
            
            # Extract answer
            pred_letter = extract_answer_letter(response)
            
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
    output_json = str(cfg.output_json)
    
    print(f"Loading model from: {model_path}")
    
    # Try to load as vision-language model first, fall back to text-only
    # transformers 5.x uses AutoModelForImageTextToText instead of AutoModelForVision2Seq
    try:
        # Attempt 1: Try as multimodal model (transformers 5.x style)
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            # transformers 4.x style
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
        # Attempt 2: Load as text-only causal LM
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
        processor = tokenizer  # Use tokenizer as processor
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
