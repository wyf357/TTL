"""Quick test to find correct Qwen3.5 multimodal input format."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import tensorflow as tf
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# Load one image from TFRecord
def load_one_image(tfrecord_path):
    def parse_example(example_proto):
        feature_description = {
            'question': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.VarLenFeature(tf.string),
            'answer': tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        parsed_features['image/encoded'] = tf.sparse.to_dense(parsed_features['image/encoded'])
        return parsed_features
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    example = next(iter(dataset.take(1)))
    
    question = example['question'].numpy().decode('utf-8')
    images_encoded = example['image/encoded'].numpy()
    
    img_tensor = tf.io.decode_image(images_encoded[0])
    img_numpy = img_tensor.numpy()
    image = Image.fromarray(img_numpy[:,:,:3], mode='RGB')
    
    return question, image

# Test different input methods
def test_methods(model_path, tfrecord_path):
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    question, image = load_one_image(tfrecord_path)
    print(f"Question: {question[:80]}...")
    print(f"Image size: {image.size}")
    
    device = torch.device("cuda")
    
    # Method 1: Direct processor with image placeholder in text
    print("\n" + "="*60)
    print("Method 1: Using <|vision_start|><|image|><|vision_end|> placeholder")
    print("="*60)
    try:
        # Qwen3.5 uses special tokens for images
        prompt = f"<|vision_start|><|image|><|vision_end|>{question}"
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        print(f"Keys: {list(inputs.keys())}")
        print(f"input_ids shape: {inputs['input_ids'].shape}")
        print(f"Has pixel_values: {'pixel_values' in inputs}")
        print(f"Has image_grid_thw: {'image_grid_thw' in inputs}")
        
        # Count image tokens
        # Qwen3.5 vision tokens are typically in a specific range
        input_ids = inputs['input_ids'].to(device)
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': inputs.get('attention_mask', None),
        }
        if 'pixel_values' in inputs:
            model_inputs['pixel_values'] = inputs['pixel_values'].to(device)
        if 'image_grid_thw' in inputs:
            model_inputs['image_grid_thw'] = inputs['image_grid_thw'].to(device)
        if 'mm_token_type_ids' in inputs:
            model_inputs['mm_token_type_ids'] = inputs['mm_token_type_ids'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"✓ SUCCESS! Response: {response[:100]}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
    
    # Method 2: Using chat template (the correct way for Qwen3.5)
    print("\n" + "="*60)
    print("Method 2: Using apply_chat_template (official way)")
    print("="*60)
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"Generated text (first 150 chars): {text[:150]}")
        
        # Process with text and images
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        
        print(f"Keys: {list(inputs.keys())}")
        print(f"input_ids shape: {inputs['input_ids'].shape}")
        print(f"Has pixel_values: {'pixel_values' in inputs}")
        if 'pixel_values' in inputs:
            print(f"pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"Has image_grid_thw: {'image_grid_thw' in inputs}")
        if 'image_grid_thw' in inputs:
            print(f"image_grid_thw: {inputs['image_grid_thw']}")
        
        # Prepare model inputs
        input_ids = inputs['input_ids'].to(device)
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': inputs.get('attention_mask', None).to(device) if inputs.get('attention_mask') is not None else None,
        }
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            model_inputs['pixel_values'] = inputs['pixel_values'].to(device)
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            model_inputs['image_grid_thw'] = inputs['image_grid_thw'].to(device)
        if 'mm_token_type_ids' in inputs and inputs['mm_token_type_ids'] is not None:
            model_inputs['mm_token_type_ids'] = inputs['mm_token_type_ids'].to(device)
        
        print(f"model_inputs keys: {list(model_inputs.keys())}")
        
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"✓ SUCCESS! Response: {response[:100]}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/Qwen3.5-2B")
    parser.add_argument("--tfrecord_path", default="./data/erqa.tfrecord")
    args = parser.parse_args()
    
    test_methods(args.model_path, args.tfrecord_path)
