#!/usr/bin/env python3
"""Deep inspect Qwen3_5ForConditionalGeneration model structure."""

import sys
import torch

model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B"

print(f"Loading model from: {model_path}")
print("=" * 80)

# Load model
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print(f"\nModel class: {model.__class__.__name__}")
print(f"Model type: {type(model)}")

# Print ALL top-level children
print("\n" + "=" * 80)
print("TOP-LEVEL CHILDREN (model.named_children()):")
for name, module in model.named_children():
    print(f"  {name}: {module.__class__.__name__}")

# Check each attribute recursively
print("\n" + "=" * 80)
print("CHECKING FOR 'layers' ATTRIBUTE:")

def find_layers(obj, path="model", depth=0, max_depth=5):
    """Recursively search for 'layers' attribute."""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    
    # Check if this object has 'layers'
    if hasattr(obj, 'layers'):
        layers = getattr(obj, 'layers')
        if isinstance(layers, torch.nn.ModuleList):
            print(f"{indent}✓ Found at: {path}.layers")
            print(f"{indent}  Type: {type(layers)}")
            print(f"{indent}  Length: {len(layers)}")
            if len(layers) > 0:
                print(f"{indent}  Layer 0 type: {type(layers[0])}")
                # Check layer 0 for input_layernorm
                layer0 = layers[0]
                for lname, lmod in layer0.named_children():
                    print(f"{indent}    Layer 0 child: {lname}: {lmod.__class__.__name__}")
    
    # Recurse into children
    if hasattr(obj, 'named_children'):
        for name, child in obj.named_children():
            find_layers(child, f"{path}.{name}", depth + 1, max_depth)

find_layers(model)

# Also check config
print("\n" + "=" * 80)
print("MODEL CONFIG:")
print(f"  model_type: {model.config.model_type}")
print(f"  hidden_size: {getattr(model.config, 'hidden_size', 'N/A')}")
print(f"  num_hidden_layers: {getattr(model.config, 'num_hidden_layers', 'N/A')}")

print("\n" + "=" * 80)
print("SUMMARY - Layer access path:")
if hasattr(model, 'vision_model') and hasattr(model, 'language_model'):
    print("  This is a vision-language model with separate vision and language components")
    if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        print("  → Use: model.language_model.model.layers")
    elif hasattr(model.language_model, 'layers'):
        print("  → Use: model.language_model.layers")
    else:
        print("  → ERROR: Cannot find layers in language_model!")
elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
    print("  → Use: model.model.layers")
elif hasattr(model, 'layers'):
    print("  → Use: model.layers")
else:
    print("  → ERROR: Cannot find layers!")
    print("\nFull attribute tree:")
    for name, module in model.named_modules():
        if 'layer' in name.lower():
            print(f"    {name}: {module.__class__.__name__}")
