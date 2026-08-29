#!/usr/bin/env python3
"""Inspect Qwen3.5 model structure to find layer access path."""

import sys
import torch
from transformers import AutoConfig

# Try to detect model type from config first
model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B"

print(f"Inspecting model from: {model_path}")
print("=" * 80)

# Load config first to determine model type
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"Model type: {config.model_type}")
print(f"Architectures: {config.architectures}")
print("=" * 80)

# Load appropriate model class based on architecture
if config.architectures and 'ConditionalGeneration' in config.architectures[0]:
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

print(f"\nModel type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")
print(f"\nTop-level attributes (nn.Module):")
for name in dir(model):
    if not name.startswith('_') and name not in ['forward', 'generate', 'config', 'tokenizer', 'processor']:
        attr = getattr(model, name)
        if isinstance(attr, torch.nn.Module):
            print(f"  {name}: {type(attr).__name__}")

# Check for language_model (multimodal)
print("\n" + "=" * 80)
if hasattr(model, 'language_model'):
    print("Found 'language_model' attribute (multimodal model):")
    print(f"  Type: {type(model.language_model)}")
    
    if hasattr(model.language_model, 'model'):
        print(f"  language_model.model type: {type(model.language_model.model)}")
        
        if hasattr(model.language_model.model, 'layers'):
            print(f"\n  ✓ Found model.language_model.model.layers!")
            print(f"  Number of layers: {len(model.language_model.model.layers)}")
            print(f"  Layer 0 type: {type(model.language_model.model.layers[0])}")
            
            layer0 = model.language_model.model.layers[0]
            print(f"\n  Layer 0 attributes:")
            for name in dir(layer0):
                if not name.startswith('_') and name not in ['forward']:
                    attr = getattr(layer0, name)
                    if isinstance(attr, torch.nn.Module):
                        print(f"    {name}: {type(attr).__name__}")
            
            if hasattr(layer0, 'input_layernorm'):
                print(f"\n  ✓ Found input_layernorm: {type(layer0.input_layernorm)}")
    elif hasattr(model.language_model, 'layers'):
        print(f"\n  ✓ Found model.language_model.layers!")
        print(f"  Number of layers: {len(model.language_model.layers)}")
            
else:
    print("✗ No 'language_model' attribute found")
    
    # Fallback: check for model.model.layers
    if hasattr(model, 'model'):
        print(f"\nChecking model.model.layers:")
        if hasattr(model.model, 'layers'):
            print(f"  ✓ Found model.model.layers!")
            print(f"  Number of layers: {len(model.model.layers)}")
        else:
            print(f"  ✗ No 'layers' in model.model")
    elif hasattr(model, 'layers'):
        print(f"  ✓ Found model.layers directly!")
        print(f"  Number of layers: {len(model.layers)}")

# Check config
print("\n" + "=" * 80)
print("Model config:")
print(f"  hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
print(f"  num_hidden_layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
print(f"  model_type: {getattr(config, 'model_type', 'N/A')}")

print("\n" + "=" * 80)
print("\nLayer access path for MANTA:")
if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
    print("  Use: model.language_model.model.layers")
elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
    print("  Use: model.model.layers")
elif hasattr(model, 'layers'):
    print("  Use: model.layers")
else:
    print("  ERROR: Cannot determine layer access path!")
