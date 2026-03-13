# Evaluation Guide

Run the evaluation harness to measure model performance.

## Quick evaluation

```bash
python scripts/evaluate.py \
    --data_dir data/test_pairs/ \
    --checkpoint checkpoints/latest \
    --mode controlnet \
    --output results/eval_report.json
```

## Metrics

### Image quality
- **FID** (Frechet Inception Distance) - measures realism of generated images
- **LPIPS** (Learned Perceptual Image Patch Similarity) - perceptual difference from ground truth
- **SSIM** (Structural Similarity) - structural similarity to ground truth

### Anatomical accuracy
- **NME** (Normalized Mean Error) - landmark displacement error normalized by inter-ocular distance

### Identity preservation
- **Identity Sim** - ArcFace cosine similarity between input and output (should stay high)

### Fairness
All metrics are automatically stratified by **Fitzpatrick skin type** (I-VI) using the ITA (Individual Typology Angle) method to ensure equitable performance across skin tones.

## Interpreting results

```json
{
  "fid": 48.3,
  "lpips_mean": 0.142,
  "ssim_mean": 0.823,
  "nme_mean": 0.041,
  "identity_sim_mean": 0.871,
  "by_fitzpatrick": {
    "I": {"lpips": 0.138, "ssim": 0.831, "nme": 0.039},
    "II": {"lpips": 0.141, "ssim": 0.825, "nme": 0.040},
    "III": {"lpips": 0.143, "ssim": 0.821, "nme": 0.042},
    "IV": {"lpips": 0.145, "ssim": 0.818, "nme": 0.043},
    "V": {"lpips": 0.148, "ssim": 0.814, "nme": 0.044},
    "VI": {"lpips": 0.151, "ssim": 0.810, "nme": 0.045}
  }
}
```

Large disparities between Fitzpatrick types indicate bias that needs to be addressed.

## Batch evaluation

```bash
# Evaluate all modes
for mode in controlnet img2img tps; do
    python scripts/evaluate.py \
        --data_dir data/test_pairs/ \
        --checkpoint checkpoints/latest \
        --mode $mode \
        --output results/eval_${mode}.json
done
```

## SLURM evaluation

```bash
# Add to your SLURM script
srun python scripts/evaluate.py \
    --data_dir data/test_pairs/ \
    --checkpoint checkpoints/latest \
    --mode controlnet \
    --output results/eval_report.json
```
