# MANTA Mask Calculation Diagnostic Guide

## Problem Analysis

**Root Cause #1: Mask Calculation Error → Loss Always Zero (40% probability)**

If `num_image_tokens` is calculated as 0, then:
- `visual_mask` is all False
- `h_v.numel() == 0`
- `_compute_iade_loss` returns `0.0` immediately
- Gamma parameters never get gradients
- Model performs exactly like baseline

## Diagnostic Steps

### Step 1: Run with Debug Logging

```bash
# Enable debug logging to see mask calculation details
python evaluations/run_erqa.py strategy=manta max_examples=5 2>&1 | grep -E "MANTA|image_token|visual_mask"
```

**Expected Output (if working correctly):**
```
[MANTA Strategy] MANTA Setup: hidden_size=2048, num_layers=24, adapt_layers=4, tau=45.25
[MANTA Strategy] MANTA initialized: 4 layers, 8 parameters, lr=0.01, lambda_global=1.0
[MANTA Strategy] Registering MANTA hooks on last 4 layers (indices 20 to 23)
[MANTA Strategy] Computed num_image_tokens=576 from 1 images (merge_size=2)
[MANTA Strategy] MANTA loss: 5.234567, gamma[0]=[1.02, 0.98]
```

**Signs of Mask Calculation Bug:**
```
[MANTA Strategy] image_grid_thw not found in inputs
[MANTA Strategy] No image tokens found! This will result in zero IADE loss.
[MANTA Strategy] MANTA loss: 0.000000, gamma[0]=[1.0, 1.0]  # gamma unchanged!
```

### Step 2: Quick Verification - Hardcode num_image_tokens

To verify if mask calculation is the issue, temporarily hardcode `num_image_tokens`:

Edit `src/openttl/strategies/manta.py`, line ~320:

```python
# TEMPORARY DEBUG: Hardcode num_image_tokens
num_image_tokens = self._compute_num_image_tokens(inputs)
STRATEGY_LOG.warning(f"DEBUG: Computed num_image_tokens={num_image_tokens}")

# FORCE a value for testing
if num_image_tokens == 0:
    STRATEGY_LOG.warning("DEBUG: Forcing num_image_tokens=256 for testing")
    num_image_tokens = 256  # Typical value for 1 image
```

Then run:
```bash
python evaluations/run_erqa.py strategy=manta max_examples=3
```

**If gamma starts changing and accuracy improves**, the mask calculation is the bug.

### Step 3: Check image_grid_thw in Batch

Add this debug code in `compute_loss` after line 317:

```python
# Debug: Check what's in inputs
STRATEGY_LOG.info(f"DEBUG: inputs keys = {list(inputs.keys())}")
if "image_grid_thw" in inputs:
    grid = inputs["image_grid_thw"]
    STRATEGY_LOG.info(f"DEBUG: image_grid_thw shape={grid.shape}, values={grid}")
else:
    STRATEGY_LOG.error("DEBUG: image_grid_thw NOT in inputs!")
```

## Possible Fixes

### Fix 1: image_grid_thw Not in Batch

**Symptom:** `image_grid_thw not found in inputs`

**Cause:** The adapter's `build_forward_inputs` may not be including `image_grid_thw` in the batch.

**Solution:** Check `src/openttl/adapters/auto.py`, line ~330:

```python
for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
    if key in prompt_enc and prompt_enc[key] is not None:
        batch[key] = prompt_enc[key]
```

This should already include `image_grid_thw`. If it's missing, the issue is earlier in the encoding pipeline.

### Fix 2: image_grid_thw Has Wrong Shape

**Symptom:** `image_grid_thw has unexpected shape: torch.Size([...])`

**Possible shapes:**
- ✅ Correct: `(num_images, 3)` where 3 = [t, h, w]
- ❌ Wrong: `(3,)` - single image without batch dim
- ❌ Wrong: `(num_images,)` - missing t,h,w dimensions

**Solution:** Handle edge cases in `_compute_num_image_tokens`:

```python
# Handle single image case
if image_grid_thw.dim() == 1 and image_grid_thw.shape[0] == 3:
    image_grid_thw = image_grid_thw.unsqueeze(0)  # (3,) -> (1, 3)
```

### Fix 3: merge_size Mismatch

**Symptom:** `num_image_tokens` is calculated but wrong value

**Qwen3.5-VL config:**
- `patch_size = 14`
- `merge_size = 2`
- Formula: `num_tokens = (t * h * w) / (merge_size^2)`

**Verify merge_size:**
```python
# In setup(), add:
if hasattr(model.config, 'vision_config'):
    merge_size = model.config.vision_config.get('merge_size', 2)
    self._merge_size = merge_size
    STRATEGY_LOG.info(f"Using merge_size={merge_size} from model config")
```

## Current Implementation

The current `_compute_num_image_tokens` implementation:

```python
def _compute_num_image_tokens(self, inputs: Dict[str, Any]) -> int:
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return 0
    
    merge_size = 2  # Default for Qwen3.5-VL
    merge_length = merge_size ** 2
    
    if image_grid_thw.dim() == 2 and image_grid_thw.shape[0] > 0:
        num_tokens = 0
        for i in range(image_grid_thw.shape[0]):
            grid = image_grid_thw[i]  # [t, h, w]
            num_tokens += int(grid.prod().item() // merge_length)
        return num_tokens
    
    return 0
```

This matches the adapter's calculation in `auto.py` line 213:
```python
num_i = int(g.prod().item() // _merge_length)
```

## Validation Checklist

Run through this checklist:

- [ ] `image_grid_thw` is present in batch inputs
- [ ] `image_grid_thw.shape` is `(num_images, 3)`
- [ ] `num_image_tokens > 0` (check logs)
- [ ] `visual_mask.sum() > 0` (check logs)
- [ ] `text_mask.sum() > 0` (check logs)
- [ ] `h_v.shape` is `(N_v, d)` where `N_v > 0`
- [ ] `h_t.shape` is `(N_t, d)` where `N_t > 0`
- [ ] IADE loss is non-zero (e.g., `5.23`)
- [ ] Gamma values change during training (e.g., `[1.0, 1.0]` → `[1.05, 0.95]`)

## Quick Test Command

```bash
# Run with verbose logging
python evaluations/run_erqa.py \
    strategy=manta \
    max_examples=3 \
    2>&1 | tee manta_debug.log

# Check for key indicators
grep -E "num_image_tokens|visual_mask|MANTA loss|gamma" manta_debug.log
```

## Expected Behavior

If everything is working:
1. Loss should be non-zero (typically 4-7 range)
2. Loss should decrease over steps
3. Gamma values should deviate from 1.0
4. Accuracy should be higher than baseline (0.4)

If loss is always 0.0 or gamma stays at [1.0, 1.0], mask calculation is broken.
