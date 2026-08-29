# MANTA-Uni Bug Fix Report

## Problem

MANTA strategy showed accuracy ~0.4 on ERQA, same as baseline (no adaptation). Investigation revealed three critical implementation bugs.

## Critical Bugs Found & Fixed

### Bug 1: Double Normalization (Fixed ✓)

**Problem:**
```python
# OLD CODE - WRONG
for l in range(L):
    hidden = self._modality_rmsnorm(hidden, l, ...)  # 1st normalization
    layer_output = layers[l](hidden)  # 2nd normalization!
```

The `layers[l](hidden)` internally calls `input_layernorm` again, causing:
- Double RMSNorm: `x → x/RMS(x)·γ₁ → (x/RMS(x)·γ₁)/RMS(...)·γ₂`
- Variance severely compressed, attention signals nearly zero
- All layers affected, even non-adaptation ones

**Fix:**
Use forward hooks to apply gamma scaling AFTER the original RMSNorm completes:

```python
# NEW CODE - CORRECT
def rmsnorm_hook(module, input, output):
    # output is already normalized by original RMSNorm
    # Just apply modality-aware gamma scaling
    scale = v_mask * gamma_v + t_mask * gamma_t
    return output * scale

rmsnorm.register_forward_hook(rmsnorm_hook)
```

### Bug 2: Optimization-Inference Disconnect (Fixed ✓)

**Problem:**
```python
# OLD CODE - WRONG
# Step 1: Manual forward with gamma (separate computation graph)
for l in range(L):
    hidden = self._modality_rmsnorm(hidden, ...)  # uses self.gamma
    hidden = layers[l](hidden)

# Step 2: Inference using standard model forward (NO gamma!)
with torch.no_grad():
    out = tta_model_forward(model, inputs)  # ignores self.gamma!
```

Gamma was optimized in Step 1 but never used in Step 2. The model generated answers using original unmodulated RMSNorm.

**Fix:**
Hooks ensure gamma modulation is applied during EVERY forward pass, including inference:

```python
# NEW CODE - CORRECT
# Hooks are registered in setup()
# When tta_model_forward is called, hooks automatically apply gamma
self._current_visual_mask = visual_mask
self._current_text_mask = text_mask

out = tta_model_forward(model, inputs)  # hooks apply gamma!
```

### Bug 3: Missing Position Encodings & KV Cache (Fixed ✓)

**Problem:**
Manual layer-by-layer forward:
```python
hidden = model.model.embed_tokens(input_ids)
for l in range(L):
    hidden = layers[l](hidden)  # Missing position_ids, attention_mask, etc.
```

This bypasses critical components:
- RoPE position encodings
- Causal attention masks
- KV cache management

**Fix:**
Use `tta_model_forward()` which properly handles all model inputs:

```python
# NEW CODE - CORRECT
from openttl.strategies.tta_shared import tta_model_forward

out = tta_model_forward(model, inputs)  # Handles everything correctly
```

## Implementation Architecture (After Fix)

```
┌─────────────────────────────────────────────────────────┐
│ setup()                                                  │
│  1. Create gamma [K, 2] parameters                      │
│  2. Register RMSNorm hooks on last K layers             │
│     └→ Hook applies gamma scaling after original norm   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ compute_loss()                                           │
│  1. Create visual/text masks from image_grid_thw        │
│  2. Register output hooks to collect hidden states      │
│  3. For each adapt_step:                                │
│     a. Clear stored outputs                             │
│     b. Forward pass (RMSNorm hooks auto-apply gamma)    │
│     c. Collect hidden states from output hooks          │
│     d. Compute IADE loss                                │
│     e. Backward + optimizer.step()                      │
│  4. Remove output hooks                                 │
│  5. Return loss + outputs                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Inference (generate)                                     │
│  - RMSNorm hooks still active → gamma applied            │
│  - Model generates with modulated representations        │
└─────────────────────────────────────────────────────────┘
```

## Key Changes Summary

| Aspect | Before (Buggy) | After (Fixed) |
|--------|---------------|---------------|
| **RMSNorm** | Manual reimplementation | Hook on original RMSNorm |
| **Normalization** | Applied twice per layer | Applied once + gamma scaling |
| **Forward Pass** | Manual layer loop | `tta_model_forward()` |
| **Gamma Usage** | Only in training loop | In all forward passes (train + inference) |
| **Position Encoding** | Missing | Handled by `tta_model_forward()` |
| **Hidden States** | Manual collection | Output hooks collect automatically |
| **Gradient Flow** | Broken (separate graphs) | Intact (single computation graph) |

## How Hooks Work

### RMSNorm Hook (Modulation)
```python
def rmsnorm_hook(module, input, output):
    """
    Called after original RMSNorm completes.
    output: (B, N, d) - already normalized
    
    Apply: output * gamma_modality
    """
    gamma_v = self.gamma[k, 0]  # visual gamma
    gamma_t = self.gamma[k, 1]  # text gamma
    
    scale = torch.ones(B, N, 1)
    scale[visual_mask] = gamma_v
    scale[text_mask] = gamma_t
    
    return output * scale  # Modulated output
```

### Output Hook (Hidden State Collection)
```python
def output_hook(module, input, output):
    """
    Called after decoder layer completes.
    Collects hidden state for IADE loss computation.
    """
    hidden = output[0] if isinstance(output, tuple) else output
    self._stored_hidden_outputs.append((layer_idx, hidden))
    return output  # Don't modify, just collect
```

## Testing Checklist

After fix, verify:

1. **Gamma values change during training:**
   ```python
   gamma_before = strategy.gamma.clone()
   trainer.train()
   gamma_after = strategy.gamma
   print(f"Gamma changed: {(gamma_before != gamma_after).any()}")
   ```

2. **Loss decreases:**
   ```
   [MANTA Strategy] MANTA loss: 5.234567
   [MANTA Strategy] MANTA loss: 4.891234
   [MANTA Strategy] MANTA loss: 4.567890
   ```

3. **Accuracy improves over baseline:**
   ```
   Baseline (no TTA): 0.40
   MANTA (fixed):     0.45+ (expected improvement)
   ```

4. **Hooks are registered:**
   ```
   [MANTA Strategy] Registering MANTA hooks on last 4 layers (indices 20 to 23)
   ```

## Algorithm-Level Considerations

Even with fixed code, MANTA's effectiveness depends on:

1. **Task Alignment:** ERQA requires spatial reasoning and action memory. IADE optimizes visual-text alignment, which may be indirect for this task.

2. **Parameter Budget:** Only 8 scalars (K=4) may be insufficient for complex sim-to-real domain shift. Try:
   ```yaml
   num_adapt_layers: 8  # 16 parameters
   adapt_steps: 3       # More optimization steps
   ```

3. **Lambda Tuning:** The global diversity weight may need adjustment:
   ```yaml
   lambda_global: 0.5  # Reduce if loss doesn't decrease
   ```

4. **Learning Rate:** Try different lr values:
   ```yaml
   lr: 5.0e-3  # or 1e-2, 2e-2
   ```

## Files Modified

- `src/openttl/strategies/manta.py` - Complete rewrite with hook-based architecture

## References

- Original algorithm: `MANTA.md`
- Hook pattern: `src/openttl/strategies/e3tta.py` (E3-TTA uses similar approach)
- TTA shared utilities: `src/openttl/strategies/tta_shared.py`
