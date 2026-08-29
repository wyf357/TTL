# MANTA-Uni Strategy Usage Guide

## Overview

MANTA-Uni (Modality-Aware Normalization Tuning with Unified Entropy) is a minimal-parameter test-time adaptation algorithm designed for early-fusion multimodal large language models.

### Key Features

- **Ultra-minimal parameters**: Only 2K scalars (e.g., 8 parameters for K=4 layers)
- **Architecture-agnostic**: Works with any early-fusion model (Qwen3.5-VL, Gemma-4V, etc.)
- **Modality-aware**: Separately modulates visual and text token representations
- **Self-supervised**: No ground truth labels required during adaptation

### Algorithm

1. **Modality-aware RMSNorm Modulation**: Apply learnable gamma scaling to visual and text tokens separately in the last K layers
2. **IADE Loss**: Minimize Implicit Cross-Modal Association Distribution Entropy to improve cross-modal alignment
3. **Forward Pass**: Custom layer-by-layer forward with RMSNorm modulation before each layer

## Quick Start

### Basic Usage on ERQA

```bash
# Run with default settings (K=4 layers)
python evaluations/run_erqa.py strategy=manta

# Run with custom parameters
python evaluations/run_erqa.py strategy=manta \
    strategy.num_adapt_layers=2 \
    strategy.lambda_global=0.5

# Limit to first 10 examples for testing
python evaluations/run_erqa.py strategy=manta max_examples=10
```

### Using the Helper Script

```bash
# Basic run
bash scripts/run_erqa_manta.sh

# With custom settings
bash scripts/run_erqa_manta.sh max_examples=10 strategy.num_adapt_layers=2

# With different model
bash scripts/run_erqa_manta.sh model=qwen35_9B max_examples=50
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_adapt_layers` | 4 | Number of last layers to adapt (K). Total parameters = 2*K |
| `lambda_global` | 1.0 | Weight for global diversity term in IADE loss |
| `lr` | 1e-2 | Learning rate for gamma parameters |
| `epsilon` | 1e-9 | Numerical stability epsilon |
| `adapt_steps` | 1 | Optimization steps per batch (increase for online TTA) |

### Parameter Recommendations

**For fast adaptation (minimal parameters)**:
```yaml
num_adapt_layers: 1  # Only 2 parameters
lr: 1e-2
```

**For balanced performance**:
```yaml
num_adapt_layers: 4  # 8 parameters (recommended)
lambda_global: 1.0
lr: 1e-2
```

**For complex scenes (more adaptation)**:
```yaml
num_adapt_layers: 8  # 16 parameters
lambda_global: 0.5
lr: 5e-3
```

## How It Works

### Visual/Text Token Separation

MANTA automatically separates visual and text tokens using `image_grid_thw` from the batch:

```python
# Token order in Qwen3.5: [visual_tokens, text_tokens]
num_image_tokens = sum(grid.prod() // merge_size**2 for each image)
visual_mask = tokens[:, :num_image_tokens]
text_mask = tokens[:, num_image_tokens:]
```

### Modality-aware RMSNorm

For each of the last K layers:

```python
# Standard RMS normalization
rms = hidden.pow(2).mean(dim=-1).sqrt()
normed = hidden / rms

# Apply modality-specific scaling
scale[visual_tokens] = gamma_v  # Learnable
scale[text_tokens] = gamma_t    # Learnable
modulated = normed * (scale * gamma_0)  # gamma_0 is frozen
```

### IADE Loss

The loss has two components:

1. **Local Peaking**: Encourage each text token to focus on specific visual tokens
   ```
   L_local = mean_i H(P_i)  # Average entropy per text token
   ```

2. **Global Diversity**: Ensure visual tokens are used diverse across all text tokens
   ```
   L_global = H(mean_i P_i)  # Entropy of average distribution
   ```

**Combined**: `L = L_local - lambda * L_global`

## Comparison with Other Strategies

| Strategy | Parameters | Loss Type | Multimodal | Speed |
|----------|-----------|-----------|------------|-------|
| TENT | LoRA params | Output entropy | ✓ | Fast |
| COME | LoRA params | Opinion entropy | ✓ | Medium |
| E3-TTA | K*d vectors | Attention entropy | ✓ | Medium |
| **MANTA** | **2K scalars** | **Cross-modal entropy** | **✓** | **Fast** |

MANTA uses the **fewest parameters** while maintaining strong cross-modal alignment.

## Troubleshooting

### "No image tokens found" warning

This occurs when processing text-only inputs. MANTA will skip adaptation and return a dummy loss. This is expected behavior.

### Shape mismatch errors

Ensure your model's `image_grid_thw` is correctly computed. MANTA relies on this to separate visual/text tokens.

### Loss not decreasing

Try adjusting:
- Increase `lr` (e.g., `5e-2`)
- Increase `adapt_steps` (e.g., `3`)
- Decrease `lambda_global` (e.g., `0.5`)

### Memory issues

Reduce `num_adapt_layers` or use gradient checkpointing:
```bash
python evaluations/run_erqa.py strategy=manta strategy.num_adapt_layers=2
```

## Advanced Usage

### Online TTA with Multiple Steps

For online test-time adaptation (per-example updates):

```bash
python evaluations/run_erqa.py \
    strategy=manta \
    strategy.adapt_steps=3 \
    online.enabled=true
```

### Compare Multiple Strategies

```bash
# TENT
python evaluations/run_erqa.py strategy=tent max_examples=100 output_json=outputs/tent_results.json

# MANTA
python evaluations/run_erqa.py strategy=manta max_examples=100 output_json=outputs/manta_results.json

# E3-TTA
python evaluations/run_erqa.py strategy=e3tta max_examples=100 output_json=outputs/e3tta_results.json
```

## Citation

If you use MANTA in your research, please cite:

```bibtex
@article{manta2024,
  title={MANTA: Modality-Aware Normalization Tuning for Test-Time Adaptation},
  author={...},
  year={2024}
}
```

## References

- Full algorithm description: `MANTA.md`
- Implementation: `src/openttl/strategies/manta.py`
- Configuration: `configs/strategy/manta.yaml`
