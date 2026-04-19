# ERQA Benchmark Evaluation Guide

## Overview

This guide explains how to use the ERQA (Embodied Reasoning Question Answer) benchmark with your local Qwen3.5-VL-2B model. The ERQA benchmark consists of 400 multimodal multiple-choice questions with interleaved images and text.

## Prerequisites

### 1. Download ERQA Dataset

Download the ERQA TFRecord file from the official repository:

```bash
# Clone the ERQA repository
git clone https://github.com/embodiedreasoning/ERQA.git
cd ERQA

# The TFRecord file is located at: data/erqa.tfrecord
# Copy it to your OpenTTL data directory
cp data/erqa.tfrecord /path/to/OpenTTL/data/erqa.tfrecord
```

### 2. Download Qwen3.5-VL-2B Model

Download the model to your AutoDL tmp directory:

```bash
# Using the provided script
python download_qwen35_2b_modelscope.py

# Or manually download to: /root/autodl-tmp/Qwen3.5-VL-2B-Instruct
```

### 3. Install Dependencies

Ensure you have the required packages:

```bash
pip install tensorflow transformers torch pillow
```

## Quick Start

### Method 1: Using the Run Script (Recommended)

```bash
# Run on full dataset (400 examples)
bash scripts/run_erqa.sh

# Run on first 10 examples (for testing)
bash scripts/run_erqa.sh max_examples=10

# Run with custom paths
bash scripts/run_erqa.sh \
  tfrecord_path=/path/to/erqa.tfrecord \
  model_path=/path/to/Qwen3.5-VL-2B-Instruct

# Use specific GPU
ERQA_GPU=1 bash scripts/run_erqa.sh max_examples=10
```

### Method 2: Using Python Directly

```bash
# Basic usage with default config
python evaluations/run_erqa.py

# Specify number of examples
python evaluations/run_erqa.py max_examples=10

# Use custom paths
python evaluations/run_erqa.py \
  tfrecord_path=/path/to/erqa.tfrecord \
  model_path=/root/autodl-tmp/Qwen3.5-VL-2B-Instruct

# With Qwen3.5-2B model config
python evaluations/run_erqa.py model=qwen35_2b max_examples=10
```

### Method 3: Using Hydra Config Override

```bash
# Change output location
python evaluations/run_erqa.py \
  output_json=my_results/erqa_eval.json \
  max_examples=50

# Override model settings
python evaluations/run_erqa.py \
  model_path=/path/to/model \
  model.torch_dtype=float16
```

## Configuration Files

### Main Config: `configs/eval_erqa.yaml`

```yaml
tfrecord_path: ./data/erqa.tfrecord      # Path to ERQA dataset
model_path: /root/autodl-tmp/Qwen3.5-VL-2B-Instruct  # Model path
max_examples: null                        # null = all 400 examples
output_json: outputs/erqa_results.json   # Output file
```

### Model Config: `configs/model/qwen35_2b.yaml`

```yaml
pretrained_model_name_or_path: /root/autodl-tmp/Qwen3.5-VL-2B-Instruct
trust_remote_code: true
attn_implementation: sdpa
torch_dtype: float16
peft:
  enabled: false  # Set true for TTA
```

## Output Format

The evaluation produces a JSON file with:

```json
{
  "metrics": {
    "accuracy": 0.75,
    "correct": 300,
    "total": 400,
    "dataset_path": "./data/erqa.tfrecord",
    "max_examples": null
  },
  "results": [
    {
      "idx": 0,
      "question": "What is the color of...",
      "answer": "A",
      "prediction": "A",
      "correct": true,
      "num_images": 1,
      "question_type": "spatial_reasoning"
    }
  ],
  "config": {
    // Full configuration used
  }
}
```

## How It Works

### Data Loading Pipeline

1. **TFRecord Parsing**: Uses TensorFlow to read `erqa.tfrecord`
2. **Feature Extraction**: Extracts:
   - `question`: Text question
   - `image/encoded`: One or more encoded images
   - `answer`: Ground truth (A, B, C, D)
   - `question_type`: Question category
   - `visual_indices`: Image placement indices

3. **Image Decoding**: Converts encoded images to PIL Image objects

### Evaluation Pipeline

1. **Model Loading**: Loads Qwen3.5-VL-2B using transformers
2. **Prompt Formatting**: Constructs multimodal prompts with images and text
3. **Inference**: Generates answer predictions
4. **Answer Extraction**: Parses model output to extract A/B/C/D
5. **Metrics Calculation**: Computes accuracy

## Advanced Usage

### Test-Time Adaptation (TTA)

If you want to use test-time adaptation:

```bash
# Enable PEFT/LoRA in config
python evaluations/run_erqa.py \
  model=qwen35_2b \
  model.peft.enabled=true \
  max_examples=10
```

Then integrate with OpenTTL's TTA strategies (TENT, COME, etc.)

### Batch Processing

For faster evaluation, you can modify the script to process in batches:

```python
# In run_erqa.py, adjust the generation parameters
outputs = model.generate(
    **model_inputs,
    max_new_tokens=50,
    do_sample=False,
    num_beams=1,  # Increase for beam search
)
```

### Custom Answer Extraction

If the default answer extraction doesn't work well, modify the `extract_answer_letter()` function in `run_erqa.py` to better match your model's output format.

## Troubleshooting

### Issue: "TensorFlow not found"

```bash
pip install tensorflow
```

### Issue: "Model not found at /root/autodl-tmp/..."

Check the model path:
```bash
ls -la /root/autodl-tmp/Qwen3.5-VL-2B-Instruct/
# Should contain: config.json, pytorch_model.bin, etc.
```

### Issue: "Out of memory"

Try:
- Reduce batch size (currently processes one example at a time)
- Use float16: `model.torch_dtype=float16`
- Use a smaller model or fewer examples

### Issue: "TFRecord file not found"

Download the ERQA dataset:
```bash
mkdir -p data/
# Download from https://github.com/embodiedreasoning/ERQA
```

## Performance Tips

1. **GPU Memory**: Qwen3.5-VL-2B requires ~5GB GPU memory in float16
2. **Speed**: Expect ~1-2 seconds per example on a modern GPU
3. **Full Evaluation**: ~10-15 minutes for all 400 examples

## File Structure

```
OpenTTL/
├── src/openttl/
│   ├── data/
│   │   ├── erqa.py                    # ERQA dataset loader
│   │   └── __init__.py                # Updated with ERQA exports
│   └── ...
├── evaluations/
│   ├── run_erqa.py                    # Main evaluation script
│   └── ...
├── configs/
│   ├── eval_erqa.yaml                 # ERQA evaluation config
│   ├── model/
│   │   └── qwen35_2b.yaml            # Qwen3.5-2B model config
│   └── ...
├── scripts/
│   └── run_erqa.sh                    # Run script
└── data/
    └── erqa.tfrecord                  # ERQA dataset (download this)
```

## References

- ERQA Benchmark: https://github.com/embodiedreasoning/ERQA
- Qwen3.5-VL: https://huggingface.co/Qwen/Qwen3.5-VL-2B-Instruct
- OpenTTL: Your existing TTA framework

## Next Steps

1. Download the ERQA dataset
2. Verify model is in the correct location
3. Run a small test: `bash scripts/run_erqa.sh max_examples=5`
4. Run full evaluation: `bash scripts/run_erqa.sh`
5. Analyze results in `outputs/erqa_results.json`
