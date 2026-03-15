# Evaluation Guide

This guide covers running the evaluation harness, understanding each metric, interpreting results, fairness analysis, and benchmarking.

## Prerequisites

Install evaluation dependencies:

```bash
pip install -e ".[eval]"
```

This pulls in `lpips`, `torch-fidelity`, `scikit-image`, and `insightface` (for ArcFace identity similarity).

## Evaluation Scripts

LandmarkDiff ships four evaluation scripts, each suited to a different workflow.

### scripts/evaluate.py: Full pipeline evaluation

Runs inference on test pairs and computes all metrics end-to-end. Use this when you want to evaluate a checkpoint against paired before/after images.

```bash
python scripts/evaluate.py \
    --test-dir data/test_pairs/ \
    --checkpoint checkpoints/phaseA/step_50000 \
    --mode controlnet \
    --output eval_results/ \
    --compute-fid \
    --compute-identity
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--test-dir` | (required) | Directory with test pairs |
| `--output` | `eval_results` | Output directory for report and images |
| `--checkpoint` | None | ControlNet checkpoint path |
| `--mode` | `tps` | Inference mode: `tps`, `controlnet`, `controlnet_ip`, `img2img` |
| `--num-samples` | 0 | Max samples to evaluate (0 = all) |
| `--compute-fid` | off | Compute FID (needs >= 50 samples) |
| `--compute-identity` | off | Compute ArcFace identity similarity (slower) |
| `--ip-adapter-scale` | 0.6 | IP-Adapter scale for controlnet_ip mode |

**Test pair directory format:**

The script supports two layouts:

```
# Format 1: Paired files with prefix
data/test_pairs/
    000001_input.png
    000001_target.png
    000002_input.png
    000002_target.png
    metadata.json        # optional: procedure, intensity per pair

# Format 2: Separate subdirectories
data/test_pairs/
    inputs/
        000001.png
        000002.png
    targets/
        000001.png
        000002.png
```

If a `metadata.json` file exists, it should map pair IDs to procedure and intensity:

```json
{
    "000001": {"procedure": "rhinoplasty", "intensity": 60},
    "000002": {"procedure": "blepharoplasty", "intensity": 50}
}
```

### scripts/evaluate_quality.py: Metric-only evaluation

Computes metrics on existing predictions without running inference. Useful when you already have generated images and just need the numbers.

```bash
# Single pair
python scripts/evaluate_quality.py \
    --pred output.png \
    --target target.png \
    --original input.png

# Batch (directories)
python scripts/evaluate_quality.py \
    --pred_dir results/predictions/ \
    --target_dir results/targets/ \
    --original_dir results/originals/ \
    --compute-fid \
    --output results/quality/
```

This produces three output files:
- `quality_results.json`: aggregate and stratified metrics
- `quality_per_sample.json`: per-image breakdown
- `quality_results.md`: markdown table for papers or PRs

### scripts/evaluate_checkpoint.py: Checkpoint evaluation

Evaluates a trained ControlNet checkpoint directly by loading the model, running generation from conditioning images, and computing all metrics. This bypasses the full LandmarkDiffPipeline and tests the ControlNet in isolation.

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/phaseA/step_50000 \
    --test_dir data/test_pairs/ \
    --output eval_results.json \
    --num_steps 20 \
    --save_images \
    --images_dir eval_images/
```

When `--save_images` is set, it saves side-by-side comparison images (conditioning | generated | target) for visual inspection.

### scripts/evaluate_on_hda.py: Clinical ground truth evaluation

Evaluates against real surgery before/after pairs from the HDA dataset. This is the clinical validation script.

```bash
python scripts/evaluate_on_hda.py \
    --checkpoint checkpoints/phaseB/best \
    --data-dir data/hda_splits/test \
    --output results/hda_eval.json \
    --num-steps 30
```

Use `--metrics-only` to skip inference and compute metrics on previously saved predictions:

```bash
python scripts/evaluate_on_hda.py \
    --metrics-only \
    --output results/hda_eval.json
```

## Understanding the Metrics

### SSIM (Structural Similarity Index)

**What it measures:** Structural similarity between the predicted and target images. Compares luminance, contrast, and structure using a sliding 11x11 Gaussian window.

**Range:** 0 to 1 (higher is better).

**Target:** > 0.80.

**Interpretation:** SSIM above 0.80 means the overall structure of the face is well preserved. Values below 0.75 suggest significant structural distortion. SSIM is sensitive to brightness and contrast shifts, so poorly matched skin tone can lower the score even when the geometry is correct.

**Implementation:** Uses scikit-image's `structural_similarity` with `channel_axis=2` for color images and `data_range=255`. Falls back to a global (non-windowed) SSIM if scikit-image is not installed, but the global version is not publication quality.

### LPIPS (Learned Perceptual Image Patch Similarity)

**What it measures:** Perceptual distance using deep features from an AlexNet backbone. Captures differences that humans notice but SSIM might miss, such as texture artifacts, blurriness, and unnatural patterns.

**Range:** 0 to 1+ (lower is better).

**Target:** < 0.15.

**Interpretation:** LPIPS below 0.10 is excellent perceptual fidelity. Between 0.10 and 0.15 is good. Above 0.20 means visible perceptual differences. LPIPS can flag issues that SSIM misses, like a face that is structurally correct but has a waxy or overly smooth texture from the diffusion model.

**Requirements:** The `lpips` package. Images are normalized to [-1, 1] range before comparison.

### NME (Normalized Mean Error)

**What it measures:** Landmark placement accuracy. Extracts MediaPipe 478-point face mesh from both the predicted and target images, computes the mean Euclidean distance between corresponding landmarks, and normalizes by inter-ocular distance (IOD, distance between landmarks 33 and 263).

**Range:** 0+ (lower is better).

**Target:** < 0.05.

**Interpretation:** NME below 0.03 means landmarks are nearly perfectly placed. Between 0.03 and 0.05 is acceptable for surgical prediction. Above 0.08 indicates significant landmark displacement, which usually means the surgical deformation was not applied correctly or the diffusion model shifted facial features.

**Note:** NME requires face detection in both images. If MediaPipe fails to detect a face in the prediction (e.g., due to artifacts), NME is reported as NaN for that sample.

### FID (Frechet Inception Distance)

**What it measures:** Distribution-level realism. Compares the distribution of Inception v3 features between real and generated image sets. Unlike the per-image metrics above, FID requires a directory of images (at least ~50 for a stable estimate).

**Range:** 0+ (lower is better).

**Target:** < 50.

**Interpretation:** FID below 30 indicates high-quality generation. Between 30 and 50 is acceptable. Above 100 suggests the model is producing unrealistic images. FID is sensitive to dataset size: with fewer than 50 images, the estimate is unstable.

**Requirements:** The `torch-fidelity` package. Runs on GPU if available, CPU otherwise (much slower).

### Identity Similarity (ArcFace)

**What it measures:** Whether the output still looks like the same person. Extracts 512-dimensional face embeddings using InsightFace's buffalo_l ArcFace model from both the input and output images, then computes cosine similarity.

**Range:** 0 to 1 (higher is better).

**Target:** > 0.85.

**Interpretation:** Above 0.90 is strong identity preservation. Between 0.80 and 0.90 is acceptable. Below 0.60 triggers the pipeline's identity drift warning. Orthognathic surgery typically has lower identity similarity because jaw repositioning changes facial proportions substantially: this is expected and the pipeline disables identity loss for orthognathic predictions during training.

**Fallback:** If InsightFace is not installed, falls back to SSIM as a rough proxy for identity similarity.

## Interpreting Results

### JSON report structure

The evaluation scripts produce a JSON report with this structure:

```json
{
    "metrics": {
        "fid": 48.3,
        "lpips": 0.142,
        "ssim": 0.823,
        "nme": 0.041,
        "identity_sim": 0.871
    },
    "config": {
        "test_dir": "data/test_pairs/",
        "checkpoint": "checkpoints/phaseA/step_50000",
        "mode": "controlnet",
        "num_samples": 200,
        "elapsed_seconds": 1234.5
    },
    "summary": "FID: 48.30\nLPIPS: 0.1420\n..."
}
```

### Per-procedure breakdown

Metrics are stratified by procedure. This matters because different procedures have different expected ranges:

| Metric | Rhinoplasty | Blepharoplasty | Rhytidectomy | Orthognathic |
|--------|------------|----------------|--------------|--------------|
| SSIM | > 0.82 | > 0.85 | > 0.78 | > 0.75 |
| LPIPS | < 0.14 | < 0.12 | < 0.16 | < 0.18 |
| NME | < 0.04 | < 0.04 | < 0.05 | < 0.06 |
| Identity | > 0.85 | > 0.88 | > 0.82 | > 0.70 |

Rhinoplasty and blepharoplasty affect smaller regions, so identity preservation is easier. Rhytidectomy covers more of the face. Orthognathic surgery changes jaw proportions, so identity scores are inherently lower.

### What to look for

1. **Large FID + good per-image metrics**: The model produces individually good images but lacks diversity or produces a narrow range of outputs. Could indicate mode collapse.

2. **Good SSIM + bad LPIPS**: The structure is correct but the texture has artifacts (waxy skin, blurred details). Common with too few diffusion steps.

3. **Good LPIPS + bad NME**: The image looks perceptually correct but the landmarks are in wrong positions. The surgical deformation was not applied properly.

4. **Low identity similarity for non-orthognathic**: The ControlNet is generating a different face. Try controlnet_ip mode or lower the `controlnet_conditioning_scale`.

## Fairness Evaluation

### Fitzpatrick skin type stratification

Every metric is automatically broken down by Fitzpatrick skin type (I through VI). This catches performance disparities across skin tones.

Skin type is classified from the input image using the Individual Typology Angle (ITA):

```
ITA = arctan((L - 50) / b) * (180 / pi)
```

where L and b come from the CIE L*a*b* color space, sampled from the center 50% of the image to avoid background pixels.

| ITA Range | Fitzpatrick Type | Description |
|-----------|-----------------|-------------|
| > 55 | Type I | Very light |
| 41 to 55 | Type II | Light |
| 28 to 41 | Type III | Intermediate |
| 10 to 28 | Type IV | Tan |
| -30 to 10 | Type V | Brown |
| < -30 | Type VI | Dark |

### Interpreting fairness results

The evaluation output includes per-type metrics:

```json
{
    "per_fitzpatrick": {
        "I":  {"ssim": {"mean": 0.831, "std": 0.02, "n": 35}},
        "II": {"ssim": {"mean": 0.825, "std": 0.03, "n": 42}},
        "III": {"ssim": {"mean": 0.821, "std": 0.02, "n": 38}},
        "IV": {"ssim": {"mean": 0.818, "std": 0.03, "n": 30}},
        "V":  {"ssim": {"mean": 0.814, "std": 0.03, "n": 25}},
        "VI": {"ssim": {"mean": 0.810, "std": 0.04, "n": 20}}
    }
}
```

**What to watch for:**

- A monotonic drop in SSIM or identity similarity from Type I to Type VI. This indicates the model performs worse on darker skin tones. The adaptive Canny edge detector (which adjusts thresholds per image) mitigates some of this, but the underlying SD1.5 model has its own biases.

- LPIPS increasing significantly for Types V and VI. This can indicate the model generates blurry or artifact-prone results on darker skin.

- Disproportionately low sample counts for certain types. If your test set has 100 Type I samples but only 5 Type VI, the Type VI numbers are not reliable.

**Mitigation strategies:**

- Ensure the training set has balanced representation across skin types.
- The LAB histogram matching in post-processing helps correct skin tone shifts.
- If a specific type has consistently worse metrics, consider type-conditioned training or type-specific hyperparameter tuning.

## Running Benchmarks

### Inference benchmarks

Measure inference speed across different hardware and modes:

```bash
python benchmarks/benchmark_inference.py --device cuda --num_images 100
```

This reports time per image and images per second for each inference mode.

### Landmark extraction benchmarks

Measure MediaPipe face mesh extraction throughput:

```bash
python benchmarks/benchmark_landmarks.py --num_images 1000
```

### Training benchmarks

Measure training steps per hour:

```bash
python benchmarks/benchmark_training.py --device cuda --num_steps 100
```

### Using Make

```bash
make evaluate   # runs default evaluation
```

### SLURM evaluation

For cluster environments, wrap the evaluation in a SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=eval-landmarkdiff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=eval_%j.log

# Skip if already completed
if [ -f eval_results/eval_report.json ]; then
    echo "Evaluation already complete, skipping."
    exit 0
fi

module load cuda/12.1
source activate landmarkdiff

python scripts/evaluate.py \
    --test-dir data/test_pairs/ \
    --checkpoint checkpoints/phaseA/step_50000 \
    --mode controlnet \
    --output eval_results/ \
    --compute-fid \
    --compute-identity
```

### Programmatic evaluation

Use the evaluation module directly in Python:

```python
import cv2
import numpy as np
from landmarkdiff.evaluation import (
    compute_ssim,
    compute_lpips,
    compute_nme,
    compute_identity_similarity,
    classify_fitzpatrick_ita,
    evaluate_batch,
)

pred = cv2.imread("prediction.png")
target = cv2.imread("target.png")

# Individual metrics
ssim = compute_ssim(pred, target)
lpips = compute_lpips(pred, target)
identity = compute_identity_similarity(pred, target)
skin_type = classify_fitzpatrick_ita(target)

print(f"SSIM: {ssim:.4f}")
print(f"LPIPS: {lpips:.4f}")
print(f"Identity: {identity:.4f}")
print(f"Fitzpatrick: Type {skin_type}")

# Batch evaluation with full stratification
metrics = evaluate_batch(
    predictions=[pred1, pred2, pred3],
    targets=[tgt1, tgt2, tgt3],
    pred_landmarks=[lm1, lm2, lm3],
    target_landmarks=[tlm1, tlm2, tlm3],
    procedures=["rhinoplasty", "rhinoplasty", "blepharoplasty"],
    compute_identity=True,
)

print(metrics.summary())

# Export to dict for JSON serialization
results_dict = metrics.to_dict()
```

## Next Steps

- [Training Guide](training.md): Train your own ControlNet checkpoint
- [Deployment Guide](deployment.md): Deploy models for production inference
- [GPU Training Guide](../GPU_TRAINING_GUIDE.md): HPC setup and multi-GPU training
