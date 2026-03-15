# Supplementary Material

Supplementary material for "LandmarkDiff: Anatomically-Conditioned Facial Surgery Outcome Prediction."

## Reproducing Supplementary Results

All tables and figures in this document can be reproduced using the scripts listed below. Run them after training is complete and a checkpoint is available at `weights/controlnet/`.

| Section | Script | Description |
|---------|--------|-------------|
| S1 (per-procedure results) | `scripts/intensity_sweep.py` | Generate per-procedure intensity sweep figures |
| S1 (per-procedure results) | `scripts/run_evaluation.py` | Compute FID/LPIPS/SSIM/NME/Identity Sim |
| S1 (per-procedure results) | `scripts/cross_procedure_eval.py` | Aggregate metrics across all 6 procedures |
| S2.1 (conditioning ablation) | `scripts/run_ablation.py` | Ablate conditioning inputs |
| S2.2 (post-processing ablation) | `scripts/run_ablation.py` | Ablate post-processing stages |
| S2.3 (RBF sensitivity) | `scripts/hyperparameter_sensitivity.py` | Sweep RBF radius |
| S2.4 (loss weight sensitivity) | `scripts/hyperparameter_sensitivity.py` | Sweep loss weights |
| S3 (Fitzpatrick analysis) | `scripts/cross_procedure_eval.py --group skin_type` | Per-skin-type breakdown |
| S4 (failure analysis) | `scripts/analyze_failures.py` | Identify and cluster failure modes |
| S4 (failure examples) | `scripts/generate_comparison_figure.py` | Render failure case comparisons |
| Denoising trajectory | `scripts/progressive_denoising.py` | Denoising strip for any procedure |
| Landmark heatmap | `scripts/landmark_accuracy_heatmap.py` | Per-landmark NME heatmap |

Quick example: reproduce S1 for rhinoplasty at all intensities:

```bash
python scripts/intensity_sweep.py \
    --input data/test_faces/ \
    --output paper/figures/supp_s1_rhinoplasty.png \
    --procedure rhinoplasty

python scripts/run_evaluation.py \
    --procedure rhinoplasty \
    --intensities 20 40 60 80 100 \
    --checkpoint weights/controlnet/ \
    --output eval/supp_s1_rhinoplasty.json
```

---

## S1. Additional Procedure Results

### S1.1 Rhinoplasty

Extended results across intensity levels (20%, 40%, 60%, 80%, 100%) on held-out test faces. See `paper/figures/fig_procedure_grid.png` for the visual grid.

| Intensity | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------|-----|-------|------|-----|-------------|
| 20% | -- | -- | -- | -- | -- |
| 40% | -- | -- | -- | -- | -- |
| 60% | -- | -- | -- | -- | -- |
| 80% | -- | -- | -- | -- | -- |
| 100% | -- | -- | -- | -- | -- |

### S1.2 Blepharoplasty

| Intensity | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------|-----|-------|------|-----|-------------|
| 20% | -- | -- | -- | -- | -- |
| 40% | -- | -- | -- | -- | -- |
| 60% | -- | -- | -- | -- | -- |
| 80% | -- | -- | -- | -- | -- |
| 100% | -- | -- | -- | -- | -- |

### S1.3 Rhytidectomy

| Intensity | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------|-----|-------|------|-----|-------------|
| 20% | -- | -- | -- | -- | -- |
| 40% | -- | -- | -- | -- | -- |
| 60% | -- | -- | -- | -- | -- |
| 80% | -- | -- | -- | -- | -- |
| 100% | -- | -- | -- | -- | -- |

### S1.4 Orthognathic

| Intensity | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------|-----|-------|------|-----|-------------|
| 20% | -- | -- | -- | -- | -- |
| 40% | -- | -- | -- | -- | -- |
| 60% | -- | -- | -- | -- | -- |
| 80% | -- | -- | -- | -- | -- |
| 100% | -- | -- | -- | -- | -- |

### S1.5 Brow Lift

| Intensity | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------|-----|-------|------|-----|-------------|
| 20% | -- | -- | -- | -- | -- |
| 40% | -- | -- | -- | -- | -- |
| 60% | -- | -- | -- | -- | -- |
| 80% | -- | -- | -- | -- | -- |
| 100% | -- | -- | -- | -- | -- |

### S1.6 Mentoplasty

| Intensity | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------|-----|-------|------|-----|-------------|
| 20% | -- | -- | -- | -- | -- |
| 40% | -- | -- | -- | -- | -- |
| 60% | -- | -- | -- | -- | -- |
| 80% | -- | -- | -- | -- | -- |
| 100% | -- | -- | -- | -- | -- |

---

## S2. Ablation Study

### S2.1 Conditioning Signal Ablation

Effect of different conditioning inputs on generation quality (rhinoplasty, 60% intensity, n=500).

| Conditioning | FID | LPIPS | SSIM | NME | Identity Sim |
|-------------|-----|-------|------|-----|-------------|
| Wireframe only | -- | -- | -- | -- | -- |
| Canny only | -- | -- | -- | -- | -- |
| Wireframe + Canny | -- | -- | -- | -- | -- |
| Full (wireframe + Canny + mask) | -- | -- | -- | -- | -- |

### S2.2 Post-Processing Ablation

Contribution of each post-processing stage (rhinoplasty, 60% intensity, n=500).

| Configuration | FID | LPIPS | SSIM | Identity Sim |
|--------------|-----|-------|------|-------------|
| Raw diffusion output | -- | -- | -- | -- |
| + CodeFormer | -- | -- | -- | -- |
| + Histogram matching | -- | -- | -- | -- |
| + Frequency-aware sharpening | -- | -- | -- | -- |
| + Laplacian pyramid blending | -- | -- | -- | -- |
| Full pipeline | -- | -- | -- | -- |

### S2.3 RBF Radius Sensitivity

Effect of Gaussian RBF influence radius on deformation smoothness and landmark accuracy.

| Radius (px) | NME | Visual Smoothness (MOS) | Artifacts |
|-------------|-----|------------------------|-----------|
| 10 | -- | -- | -- |
| 20 | -- | -- | -- |
| 30 (default) | -- | -- | -- |
| 40 | -- | -- | -- |
| 50 | -- | -- | -- |

### S2.4 Loss Weight Sensitivity

Phase B training with varied loss weights (all other weights held at default).

| Varied Weight | Value | FID | LPIPS | NME | Identity Sim |
|--------------|-------|-----|-------|-----|-------------|
| Landmark (default 0.1) | 0.01 | -- | -- | -- | -- |
| | 0.05 | -- | -- | -- | -- |
| | 0.1 | -- | -- | -- | -- |
| | 0.2 | -- | -- | -- | -- |
| Identity (default 0.05) | 0.01 | -- | -- | -- | -- |
| | 0.05 | -- | -- | -- | -- |
| | 0.1 | -- | -- | -- | -- |
| Perceptual (default 0.1) | 0.05 | -- | -- | -- | -- |
| | 0.1 | -- | -- | -- | -- |
| | 0.2 | -- | -- | -- | -- |

---

## S3. Fitzpatrick Skin Type Analysis

Per-skin-type performance breakdown to verify equitable quality across skin tones. Skin type is classified automatically via Individual Typology Angle (ITA) computed from the LAB color space of the input photo.

### S3.1 Aggregate Metrics by Skin Type

| Fitzpatrick Type | ITA Range | n | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------------|-----------|---|-----|-------|------|-----|-------------|
| Type I (very light) | > 55 | -- | -- | -- | -- | -- | -- |
| Type II (light) | 41-55 | -- | -- | -- | -- | -- | -- |
| Type III (intermediate) | 28-41 | -- | -- | -- | -- | -- | -- |
| Type IV (tan) | 10-28 | -- | -- | -- | -- | -- | -- |
| Type V (brown) | -30 to 10 | -- | -- | -- | -- | -- | -- |
| Type VI (dark) | < -30 | -- | -- | -- | -- | -- | -- |

### S3.2 Per-Procedure x Per-Skin-Type (Rhinoplasty)

| Fitzpatrick Type | FID | LPIPS | SSIM | NME | Identity Sim |
|-----------------|-----|-------|------|-----|-------------|
| Type I | -- | -- | -- | -- | -- |
| Type II | -- | -- | -- | -- | -- |
| Type III | -- | -- | -- | -- | -- |
| Type IV | -- | -- | -- | -- | -- |
| Type V | -- | -- | -- | -- | -- |
| Type VI | -- | -- | -- | -- | -- |

Additional per-procedure tables to be generated once evaluation is complete.

---

## S4. Failure Case Analysis

### S4.1 Common Failure Modes

| Failure Mode | Frequency | Cause | Mitigation |
|-------------|-----------|-------|------------|
| Identity drift at high intensity | -- | Large deformations push ControlNet output away from input identity | ArcFace verification flags; cap intensity at 80% for clinical use |
| Jaw artifacts in orthognathic | -- | Mandibular landmarks cross jawline boundary under large advancement | Identity loss disabled for orthognathic; wider RBF radius |
| Asymmetric blending seam | -- | Surgical mask boundary intersects a high-contrast region (hairline, ear) | Perlin boundary noise on mask; wider feather radius |
| Skin tone shift in shadows | -- | LAB histogram matching overcorrects in low-luminance regions | Mask-weighted matching restricted to well-lit facial area |
| Landmark detection failure | -- | Extreme head pose (yaw > 60 degrees) or heavy occlusion | Face view classifier rejects non-frontal inputs with warning |

### S4.2 Qualitative Failure Examples

To be populated with representative failure images after evaluation. Each example will include the input image, predicted output, and a brief explanation of what went wrong.

---

## S5. Computational Requirements

### S5.1 Inference

| Component | VRAM | Time (A100) | Time (RTX 4090) | Time (CPU) |
|-----------|------|-------------|-----------------|------------|
| MediaPipe landmark extraction | < 100 MB | ~10 ms | ~15 ms | ~30 ms |
| Gaussian RBF deformation | negligible | < 1 ms | < 1 ms | < 1 ms |
| TPS warp | < 200 MB | ~50 ms | ~80 ms | ~200 ms |
| ControlNet + SD 1.5 (30 steps) | ~4 GB | ~3 s | ~5 s | N/A |
| CodeFormer | ~400 MB | ~100 ms | ~200 ms | ~2 s |
| Real-ESRGAN | ~200 MB | ~50 ms | ~100 ms | ~3 s |
| Laplacian blend + compositing | negligible | ~20 ms | ~30 ms | ~50 ms |
| ArcFace verification | ~300 MB | ~15 ms | ~25 ms | ~100 ms |
| **Total (ControlNet mode)** | **~5.2 GB** | **~3.5 s** | **~5.5 s** | **N/A** |
| **Total (TPS mode)** | **< 300 MB** | **~0.3 s** | **~0.4 s** | **~0.5 s** |

### S5.2 Training

| Configuration | Hardware | Batch Size | Grad Accum | Effective Batch | Steps/hr | Time to 50K steps |
|--------------|----------|-----------|------------|-----------------|----------|-------------------|
| Phase A (synthetic) | A100 80GB | 4 | 4 | 16 | ~600 | ~83 hr |
| Phase A (synthetic) | A100 40GB | 2 | 8 | 16 | ~400 | ~125 hr |
| Phase A (synthetic) | RTX 4090 | 2 | 8 | 16 | ~350 | ~143 hr |
| Phase A (synthetic) | RTX 3090 | 1 | 16 | 16 | ~200 | ~250 hr |

### S5.3 Data Generation

| Task | Hardware | Throughput |
|------|----------|-----------|
| Synthetic pair generation (TPS) | CPU (8-core) | ~200 pairs/min |
| Synthetic pair generation (TPS) | CPU (32-core) | ~700 pairs/min |
| Landmark extraction (batch) | CPU | ~30 images/sec |
| Evaluation (full suite) | A100 | ~50 images/min |
