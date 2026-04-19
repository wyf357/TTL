"""Test script to verify ERQA data loading and Qwen3.5 processing."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import tensorflow as tf
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


def test_tfrecord_loading(tfrecord_path: str, num_examples: int = 2):
    """Test TFRecord loading and parsing."""
    print("=" * 60)
    print("TEST 1: TFRecord Loading")
    print("=" * 60)
    
    def parse_example(example_proto):
        feature_description = {
            'question': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.VarLenFeature(tf.string),
            'answer': tf.io.FixedLenFeature([], tf.string),
            'question_type': tf.io.VarLenFeature(tf.string),
            'visual_indices': tf.io.VarLenFeature(tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        parsed_features['image/encoded'] = tf.sparse.to_dense(parsed_features['image/encoded'])
        parsed_features['question_type'] = tf.sparse.to_dense(parsed_features['question_type'])
        parsed_features['visual_indices'] = tf.sparse.to_dense(parsed_features['visual_indices'])
        return parsed_features
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    
    for i, example in enumerate(dataset.take(num_examples)):
        print(f"\n--- Example {i+1} ---")
        question = example['question'].numpy().decode('utf-8')
        answer = example['answer'].numpy().decode('utf-8')
        images_encoded = example['image/encoded'].numpy()
        question_type_tensors = example['question_type']
        question_type = ""
        if len(question_type_tensors) > 0:
            question_type = question_type_tensors[0].numpy().decode('utf-8')
        visual_indices = example['visual_indices'].numpy()
        
        print(f"Question: {question[:100]}...")
        print(f"Answer: {answer}")
        print(f"Question Type: {question_type}")
        print(f"Number of images: {len(images_encoded)}")
        print(f"Visual indices: {visual_indices}")
        
        # Decode first image
        if len(images_encoded) > 0:
            img_tensor = tf.io.decode_image(images_encoded[0])
            img_numpy = img_tensor.numpy()
            print(f"First image shape: {img_numpy.shape}")
            print(f"First image dtype: {img_numpy.dtype}")


def test_qwen35_processing(model_path: str, tfrecord_path: str):
    """Test Qwen3.5 model processing with ERQA data."""
    print("\n" + "=" * 60)
    print("TEST 2: Qwen3.5 Model Processing")
    print("=" * 60)
    
    # Load model and processor
    print(f"\nLoading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Model loaded successfully!")
    
    # Load one example from TFRecord
    def parse_example(example_proto):
        feature_description = {
            'question': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.VarLenFeature(tf.string),
            'answer': tf.io.FixedLenFeature([], tf.string),
            'question_type': tf.io.VarLenFeature(tf.string),
            'visual_indices': tf.io.VarLenFeature(tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        parsed_features['image/encoded'] = tf.sparse.to_dense(parsed_features['image/encoded'])
        parsed_features['question_type'] = tf.sparse.to_dense(parsed_features['question_type'])
        parsed_features['visual_indices'] = tf.sparse.to_dense(parsed_features['visual_indices'])
        return parsed_features
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    example = next(iter(dataset.take(1)))
    
    # Decode data
    question = example['question'].numpy().decode('utf-8')
    answer = example['answer'].numpy().decode('utf-8')
    images_encoded = example['image/encoded'].numpy()
    
    print(f"\nQuestion: {question[:100]}...")
    print(f"Answer: {answer}")
    print(f"Number of images: {len(images_encoded)}")
    
    # Decode images to PIL
    from PIL import Image
    images = []
    for j, img_encoded in enumerate(images_encoded):
        img_tensor = tf.io.decode_image(img_encoded)
        img_numpy = img_tensor.numpy()
        if img_numpy.shape[-1] == 3:
            img_pil = Image.fromarray(img_numpy, mode='RGB')
        elif img_numpy.shape[-1] == 4:
            img_pil = Image.fromarray(img_numpy, mode='RGBA')
        elif img_numpy.shape[-1] == 1:
            img_pil = Image.fromarray(img_numpy.squeeze(), mode='L')
            img_pil = img_pil.convert('RGB')
        else:
            img_pil = Image.fromarray(img_numpy[:,:,:3], mode='RGB')
        images.append(img_pil)
        print(f"Image {j+1}: {img_pil.size}, mode={img_pil.mode}")
    
    # Test processor
    print("\n--- Testing Processor ---")
    
    # Qwen3.5 requires special image token format
    # The processor should handle this automatically with proper prompt format
    prompt = f"Please answer the following question based on the image:\n\n{question}"
    
    # Try using chat template format (recommended for Qwen3.5)
    print("\n1. Testing with images (chat format):")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Use apply_chat_template for proper formatting
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(f"   Formatted text (first 200 chars): {text[:200]}")
    
    inputs_with_images = processor(
        text=[text],
        images=images,
        return_tensors="pt"
    )
    print(f"   Keys: {list(inputs_with_images.keys())}")
    print(f"   input_ids shape: {inputs_with_images['input_ids'].shape}")
    if 'pixel_values' in inputs_with_images:
        print(f"   pixel_values shape: {inputs_with_images['pixel_values'].shape}")
        print(f"   pixel_values is None: {inputs_with_images['pixel_values'] is None}")
    else:
        print("   WARNING: pixel_values NOT in inputs!")
    if 'image_grid_thw' in inputs_with_images:
        print(f"   image_grid_thw: {inputs_with_images['image_grid_thw']}")
    else:
        print("   WARNING: image_grid_thw NOT in inputs!")
    
    # Check if image tokens are in input_ids
    # Count image tokens (Qwen3.5 uses specific token IDs for images)
    input_ids = inputs_with_images['input_ids']
    print(f"   input_ids sample: {input_ids[0, :20]}")
    
    print("\n2. Testing without images:")
    inputs_without_images = processor(
        text=prompt,
        return_tensors="pt"
    )
    print(f"   Keys: {list(inputs_without_images.keys())}")
    print(f"   input_ids shape: {inputs_without_images['input_ids'].shape}")
    
    # Test generation
    print("\n--- Testing Generation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n1. Generation WITH images (chat format):")
    try:
        input_ids = inputs_with_images['input_ids'].to(device)
        attention_mask = inputs_with_images.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if 'pixel_values' in inputs_with_images:
            pv = inputs_with_images['pixel_values']
            if pv is not None:
                model_inputs['pixel_values'] = pv.to(device)
                print(f"   Added pixel_values to model_inputs")
            else:
                print(f"   WARNING: pixel_values is None, skipping!")
        
        if 'image_grid_thw' in inputs_with_images:
            igt = inputs_with_images['image_grid_thw']
            if igt is not None:
                model_inputs['image_grid_thw'] = igt.to(device)
                print(f"   Added image_grid_thw to model_inputs")
            else:
                print(f"   WARNING: image_grid_thw is None, skipping!")
        
        if 'mm_token_type_ids' in inputs_with_images:
            mtt = inputs_with_images['mm_token_type_ids']
            if mtt is not None:
                model_inputs['mm_token_type_ids'] = mtt.to(device)
                print(f"   Added mm_token_type_ids to model_inputs")
        
        print(f"   model_inputs keys: {list(model_inputs.keys())}")
        print(f"   input_ids shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"   SUCCESS! Response: {response[:100]}")
        
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Generation WITHOUT images:")
    try:
        input_ids = inputs_without_images['input_ids'].to(device)
        model_inputs = {'input_ids': input_ids}
        
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        response = processor.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"   SUCCESS! Response: {response[:100]}")
        
    except Exception as e:
        print(f"   FAILED: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_path", type=str, default="./data/erqa.tfrecord")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen3.5-2B")
    parser.add_argument("--test", type=str, choices=["tfrecord", "model", "both"], default="both")
    args = parser.parse_args()
    
    if args.test in ["tfrecord", "both"]:
        test_tfrecord_loading(args.tfrecord_path, num_examples=2)
    
    if args.test in ["model", "both"]:
        test_qwen35_processing(args.model_path, args.tfrecord_path)
