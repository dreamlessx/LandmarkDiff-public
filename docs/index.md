# LandmarkDiff

[![CI](https://github.com/dreamlessx/LandmarkDiff-public/actions/workflows/ci.yml/badge.svg)](https://github.com/dreamlessx/LandmarkDiff-public/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dreamlessx/LandmarkDiff-public/blob/main/notebooks/quickstart.ipynb)

Photorealistic facial surgery outcome prediction from standard clinical photography, powered by anatomically-conditioned latent diffusion.

LandmarkDiff takes a patient's pre-operative photograph and a specified surgical procedure, then generates a photorealistic prediction of what that patient will look like post-operatively. The entire pipeline runs on a single 2D photo - no 3D CT scans, no depth sensors, no multi-view capture required.

It works by extracting MediaPipe's 478-point face mesh from the input photo, applying procedure-specific Gaussian RBF deformations calibrated from anthropometric surgical data, rendering the deformed mesh as a tessellation wireframe, and feeding that wireframe into a ControlNet-conditioned Stable Diffusion 1.5 backbone to synthesize the predicted face. The output is composited back onto the original image using Laplacian pyramid blending with feathered surgical masks, then refined through neural face restoration and identity verification.

> **Paper:** "LandmarkDiff: Anatomically-Conditioned Latent Diffusion for Photorealistic Facial Surgery Outcome Prediction," targeting MICCAI 2026.

---

## Quick Links

- [Quickstart notebook](../examples/quickstart.ipynb) -- load the pipeline, run predictions, compare procedures
- [CHANGELOG](../CHANGELOG.md) -- what changed between releases
- [Discussions](https://github.com/dreamlessx/LandmarkDiff-public/discussions) -- questions, ideas, and community discussion
- [API reference](api/) -- per-module documentation

---

## Table of Contents

**Getting started**
- [Why LandmarkDiff](#why-landmarkdiff)
- [Quick Start](#quick-start)
- [Inference Modes](#inference-modes)
- [Gradio Web Demo](#gradio-web-demo)

**Core pipeline**
- [Supported Procedures](#supported-procedures)
- [How It Works](#how-it-works)
- [Clinical Edge Cases](#clinical-edge-cases)
- [Post-Processing Pipeline](#post-processing-pipeline)

**Training and evaluation**
- [Training](#training)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Benchmarks](#benchmarks)
- [Model Zoo](#model-zoo)

**Reference**
- [Demo Outputs](#demo-outputs)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Docker](#docker)
- [Roadmap](#roadmap)

**Community**
- [Citation](#citation)
- [Contributors](#contributors)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Why LandmarkDiff

Patients considering facial surgery want to see what they'll look like afterward. Current approaches either require expensive 3D imaging hardware (structured light, CT, MRI) or produce cartoonish 2D warps that don't look real. LandmarkDiff bridges that gap:

- **2D input only.** Works from any standard clinical photograph or even a phone selfie. No 3D scanners, no depth sensors, no multi-view rigs.
- **Photorealistic output.** Diffusion-based generation produces natural skin texture, lighting, and shadows - not just a geometric morph.
- **Anatomically grounded.** Deformations are driven by procedure-specific landmark displacements calibrated from anthropometric surgical literature, not arbitrary pixel pushing.
- **Identity preserving.** ArcFace verification ensures the output still looks like the same patient, not a different person.
- **Clinically aware.** Built-in handling for vitiligo, Bell's palsy, keloid-prone skin, and Ehlers-Danlos syndrome - conditions that affect how facial tissue responds to surgery.
- **Fair across skin tones.** All evaluation metrics are stratified by Fitzpatrick skin type (I through VI) to catch and prevent performance disparities.

---

## Supported Procedures

LandmarkDiff ships with four procedure presets, each targeting specific anatomical regions with calibrated displacement vectors.

### Rhinoplasty (Nose Reshaping)

Targets 24 landmarks across the nasal bridge, tip, and alar base. Key deformations include alar base narrowing (nostril width reduction), tip refinement with upward rotation, and dorsal hump reduction. Uses a 30px Gaussian RBF influence radius for smooth transitions across the nasal region.

**Landmark indices:** 1, 2, 4, 5, 6, 19, 94, 141, 168, 195, 197, 236, 240, 274, 275, 278, 279, 294, 326, 327, 360, 363, 370, 456, 460

### Blepharoplasty (Eyelid Surgery)

Targets 28 landmarks around the upper and lower eyelids. Deformations include upper lid elevation (hooded eye correction), medial and lateral canthal tapering, and lower lid tightening. Uses a tighter 15px influence radius to avoid affecting surrounding structures like the brow.

**Landmark indices:** 33, 7, 163, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398

### Rhytidectomy (Facelift)

Targets 32 landmarks along the jawline, cheeks, and periauricular region. Deformations include jowl lifting (upward and lateral traction), submental tightening, and gentle temple lifting to simulate tissue redistribution. Uses a wider 40px influence radius for the broad soft tissue mobilization typical of facelifts.

**Landmark indices:** 10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 150, 162, 172, 176, 187, 207, 213, 234, 284, 297, 323, 332, 338, 356, 361, 365, 379, 389, 397, 400, 427, 454

### Orthognathic Surgery (Jaw Repositioning)

Targets 47 landmarks across the mandible, maxilla, and chin. Deformations simulate mandibular advancement or setback, chin projection changes, and lateral jaw narrowing. Uses a 35px influence radius. Note that identity loss is disabled for orthognathic predictions because jaw repositioning inherently changes facial proportions more than the other procedures.

**Landmark indices:** 0, 17, 18, 36, 37, 39, 40, 57, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 167, 169, 170, 175, 181, 191, 200, 201, 202, 204, 208, 211, 212, 214, 269, 270, 291, 311, 312, 317, 321, 324, 325, 375, 396, 405, 407, 415

### Adding Your Own Procedure

You can define custom procedures by specifying which landmarks to move, how far, and in what direction. See [tutorials/custom_procedures.md](tutorials/custom_procedures.md) for a step-by-step guide.

---

## How It Works

LandmarkDiff is a five-stage pipeline. Each stage is independently testable and swappable.

```
                         Input Photo (512x512)
                                |
                    [1] MediaPipe Face Mesh
                    478 landmarks, 3D coords
                                |
                    [2] Gaussian RBF Deformation
                    Procedure-specific displacement
                    vectors scaled by intensity (0-100%)
                                |
                    [3] Conditioning Generation
                    2556-edge tessellation wireframe
                    + adaptive Canny edges + mask
                                |
                    [4] ControlNet + Stable Diffusion 1.5
                    CrucibleAI/ControlNetMediaPipeFace
                    conditioned latent space generation
                                |
                    [5] Post-Processing
                    Laplacian pyramid blend (6 levels)
                    + CodeFormer face restoration
                    + Real-ESRGAN background upscale
                    + ArcFace identity verification
                                |
                        Predicted Post-Op Face
```

### Stage 1: Landmark Extraction

MediaPipe Face Mesh detects 478 facial landmarks in 3D (x, y, z normalized coordinates) at roughly 30 fps on CPU. The landmarks are grouped into anatomical regions:

| Region | Landmark count |
|--------|---------------|
| Jawline | 33 |
| Left eye | 16 |
| Right eye | 16 |
| Left eyebrow | 10 |
| Right eyebrow | 10 |
| Nose | 25 |
| Lips | 22 |
| Left iris | 5 |
| Right iris | 5 |
| Face oval | 37 |

The extraction runs at the start of every prediction and again on the output for evaluation (NME metric).

### Stage 2: Gaussian RBF Deformation

Each procedure preset defines a set of `DeformationHandle` objects, each specifying:
- **Which landmark** to move (index into the 478-point mesh)
- **How far** to move it (pixel displacement vector, scaled by the intensity slider)
- **How wide** the influence is (Gaussian RBF radius in pixels)

The deformation is applied as a smooth, spatially weighted field. Landmarks near the handle move the most; landmarks far away are unaffected. This prevents the jarring discontinuities you get from simple point-to-point warping.

All displacement magnitudes are scaled by the `intensity` parameter (0 to 100), so you can preview subtle through aggressive versions of the same procedure.

### Stage 3: Conditioning Generation

The deformed landmarks are rendered into conditioning images for ControlNet:

1. **Tessellation wireframe** - The full 2556-edge MediaPipe face mesh drawn on a black canvas. This is the primary conditioning signal. It uses a static anatomical adjacency list (not Delaunay triangulation), so the topology is invariant to landmark displacement.

2. **Adaptive Canny edges** - Edge detection with thresholds derived from the image median (low = 0.66 * median, high = 1.33 * median). This adapts to different skin tones without hardcoded thresholds, plus morphological skeletonization to produce 1-pixel edges that ControlNet expects.

3. **Surgical mask** - A feathered mask indicating where the procedure affects the face. Built from the convex hull of procedure-specific landmarks, dilated, Gaussian-feathered, then perturbed with Perlin-style boundary noise (2-4px) to prevent visible seam lines.

### Stage 4: Diffusion Generation

The conditioning images are fed to CrucibleAI's pre-trained ControlNet for MediaPipe Face, which conditions Stable Diffusion 1.5 to generate a face matching the deformed mesh topology. Procedure-specific text prompts emphasize clinical photography qualities (natural appearance, sharp focus, studio lighting).

### Stage 5: Post-Processing

Six-step refinement:
1. **CodeFormer** neural face restoration (fidelity weight 0.7 for quality-fidelity balance)
2. **Real-ESRGAN** background super-resolution (non-face regions only)
3. **Histogram matching** in LAB color space for robust skin tone transfer from input to output
4. **Frequency-aware sharpening** on the L channel only (avoids color fringing)
5. **Laplacian pyramid blending** (6 levels) - low frequencies blend smoothly for lighting continuity, high frequencies transition sharply for texture/pore preservation
6. **ArcFace identity verification** - flags if the output drifts too far from the input identity (cosine similarity threshold 0.6)

---

## Demo Outputs

Sample outputs are in the [demos/](demos/) directory.

**Pipeline visualization** (`pipeline_abstract.png`):

Schematic overview of the five-stage pipeline: Input, 478-point mesh extraction, RBF deformation, ControlNet + SD1.5 synthesis, and predicted result.

**Mesh deformation** (`mesh_deformation.png`):

Side-by-side comparison of original and deformed face meshes, showing procedure-specific Gaussian RBF displacement vectors.

ControlNet-generated photorealistic samples will be added after model training completes.

---

## Quick Start

### Installation

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public

# Core (inference only)
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With Gradio demo
pip install -e ".[app]"

# With evaluation metrics
pip install -e ".[eval]"

# Everything
pip install -e ".[train,eval,app,dev]"
```

### Run a single prediction

```bash
python scripts/run_inference.py /path/to/face.jpg \
    --procedure rhinoplasty \
    --intensity 60 \
    --mode controlnet
```

This will:
1. Detect the face and extract 478 landmarks
2. Apply rhinoplasty deformation at 60% intensity
3. Generate the ControlNet-conditioned prediction
4. Composite the result back onto the original
5. Save the output to `output/result.png`

### CPU-only mode (no GPU needed)

```bash
python examples/tps_only.py /path/to/face.jpg \
    --procedure rhinoplasty \
    --intensity 60
```

TPS mode does pure geometric warping. It runs instantly on CPU and produces a geometrically accurate result, but without the photorealistic texture synthesis that the diffusion modes provide.

### Batch processing

```bash
python examples/batch_inference.py /path/to/image_dir/ \
    --procedure blepharoplasty \
    --intensity 50 \
    --output output/batch/
```

---

## Inference Modes

LandmarkDiff supports four inference modes with different quality-speed-hardware tradeoffs:

| Mode | GPU Required | Speed | Quality | Identity Preservation |
|------|-------------|-------|---------|----------------------|
| `tps` | No | Instant (~0.5s) | Geometric only | Perfect (pixel-level) |
| `img2img` | Yes (6GB) | ~5s | Good | Good |
| `controlnet` | Yes (6GB) | ~5s | Best | Good |
| `controlnet_ip` | Yes (8GB) | ~7s | Best | Best |

**TPS mode** - Thin-plate spline warping. No diffusion, no neural network inference. Just mathematically warps the pixels according to landmark displacements. Fast and deterministic, but the output looks like a geometric morph rather than a natural photo. Good for previewing the deformation before committing to a full diffusion run.

**img2img mode** - Standard Stable Diffusion img2img with the TPS-warped image as input and a feathered mask restricting generation to the surgical region. Faster than ControlNet but less controllable.

**ControlNet mode** - The primary mode. Uses CrucibleAI's pre-trained ControlNet for MediaPipe Face mesh conditioning. The deformed wireframe directly controls the spatial layout of the generated face, producing the most anatomically accurate results.

**ControlNet + IP-Adapter mode** - Adds IP-Adapter FaceID on top of ControlNet for stronger identity preservation. Uses face embeddings from the input photo to condition generation, reducing the chance of producing a different-looking person. Slightly slower due to the additional encoder pass.

```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()

result = pipeline.generate(
    image,
    procedure="rhinoplasty",
    intensity=60,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    strength=0.75,
    seed=42,
    postprocess=True,
)

# result dict contains:
# result["output"]              - final composited image
# result["output_raw"]          - raw diffusion output (before compositing)
# result["output_tps"]          - TPS-only geometric warp
# result["conditioning"]        - wireframe fed to ControlNet
# result["mask"]                - surgical mask
# result["landmarks_original"]  - input landmarks
# result["landmarks_manipulated"] - deformed landmarks
# result["identity_check"]      - ArcFace similarity score
```

---

## Gradio Web Demo

```bash
python scripts/app.py
# Opens at http://localhost:7860
```

The demo has five tabs:

### Tab 1: Single Procedure
Upload a photo, pick a procedure, adjust intensity from 0-100%. The interface shows every intermediate step: extracted landmarks, deformed mesh, wireframe conditioning, surgical mask, TPS warp, and the final result in a side-by-side before/after view. Clinical flags (vitiligo, Bell's palsy with side selector, keloid-prone regions, Ehlers-Danlos) are available as checkboxes.

### Tab 2: Multi-Procedure Comparison
Set independent intensity sliders for all four procedures and generate them all from the same photo. Useful for showing a patient their options side by side.

### Tab 3: Intensity Sweep
Pick a procedure and a number of steps (3 to 10). Generates a gallery progressing from 0% to 100% intensity so you can see exactly how the result changes with the intensity parameter.

### Tab 4: Face Analysis
Upload a photo and get back the detected Fitzpatrick skin type, face view classification (frontal, three-quarter, or profile), yaw and pitch angles in degrees, per-region landmark counts, confidence scores, and an annotated landmark visualization.

### Tab 5: Multi-Angle Capture
Guides the user through capturing 5 standardized clinical views: frontal (0 degrees), left three-quarter (45 degrees), right three-quarter (45 degrees), left profile (90 degrees), right profile (90 degrees). Validates each photo against the expected yaw range and generates predictions for all views, producing a combined before/after gallery.

---

## Training

Training happens in two phases.

### Phase A: Synthetic Data (current)

Generate TPS-warped face pairs from FFHQ, then fine-tune ControlNet to reconstruct the original face from the deformed wireframe.

```bash
# 1. Download FFHQ samples
python scripts/download_ffhq.py --num 50000 --resolution 512

# 2. Generate training pairs (original + TPS-warped + wireframe)
python scripts/generate_synthetic_data.py \
    --input data/ffhq_samples/ \
    --output data/synthetic_pairs/ \
    --num 50000

# 3. Train ControlNet
python scripts/train_controlnet.py \
    --data_dir data/synthetic_pairs/ \
    --output_dir checkpoints/ \
    --num_train_steps 50000
```

Phase A uses diffusion loss only (MSE between predicted and target noise).

### Phase B: Clinical + Combined Loss (planned)

Fine-tune further on clinical before/after pairs with the full four-term loss:

| Loss | Weight | Purpose |
|------|--------|---------|
| Diffusion (MSE) | 1.0 | Primary training signal |
| Landmark L2 | 0.1 | Anatomical accuracy (inside surgical mask only) |
| Identity (ArcFace) | 0.05 | Patient identity preservation |
| Perceptual (LPIPS) | 0.1 | Texture quality (outside mask, prevents penalizing the TPS warp) |

The landmark loss is normalized by inter-ocular distance (landmarks 33 vs 263) for scale invariance. The identity loss uses procedure-dependent face cropping - rhinoplasty crops to the upper face, blepharoplasty uses the full face, rhytidectomy crops above the jawline, and orthognathic disables identity loss entirely since jaw surgery inherently changes proportions.

### Training Configuration

Default config at `configs/training.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-5 | With cosine scheduler |
| Warmup steps | 500 | |
| Batch size | 4 | Gradient accumulation 4x, effective batch 16 |
| Mixed precision | bf16 | NOT fp16 - activation range exceeded |
| EMA decay | 0.9999 | |
| Checkpoint interval | 5000 steps | |
| ControlNet scale max | 1.2 | Sum > 1.2 causes saturation |

Important training safeguards:
- VAE is always frozen (gradient leak corrupts the latent space)
- GroupNorm instead of BatchNorm (batch size 4 makes BN unstable)
- TPS warps are precomputed to avoid CPU bottleneck during training
- Git LFS required for checkpoints

### SLURM (HPC)

```bash
sbatch scripts/train_slurm.sh
```

See [docs/GPU_TRAINING_GUIDE.md](docs/GPU_TRAINING_GUIDE.md) for detailed HPC setup, Apptainer containers, and multi-node configurations.

---

## Evaluation and Metrics

### Primary Metrics

| Metric | What it measures | Target | How it's computed |
|--------|-----------------|--------|-------------------|
| FID | Realism | < 50 | Frechet Inception Distance via torch-fidelity (GPU-accelerated) |
| LPIPS | Perceptual similarity | < 0.15 | Learned Perceptual Image Patch Similarity (AlexNet backbone) |
| SSIM | Structural similarity | > 0.80 | Structural Similarity Index between input and output |
| NME | Landmark accuracy | < 0.05 | Normalized Mean Error - L2 distance between predicted and target landmarks, normalized by inter-ocular distance (landmarks 33 vs 263) |
| Identity Sim | Identity preservation | > 0.85 | ArcFace cosine similarity between input and output face embeddings (InsightFace buffalo_l, 512-dim) |

### Fitzpatrick Stratification

Every metric is broken down by Fitzpatrick skin type to ensure equitable performance. Skin type is classified automatically from the input photo using Individual Typology Angle (ITA):

```
ITA = arctan((L - 50) / b) * (180 / pi)
```

where L and b come from the LAB color space.

| ITA Range | Fitzpatrick Type | Description |
|-----------|-----------------|-------------|
| > 55 | Type I | Very light |
| 41 to 55 | Type II | Light |
| 28 to 41 | Type III | Intermediate |
| 10 to 28 | Type IV | Tan |
| -30 to 10 | Type V | Brown |
| < -30 | Type VI | Dark |

This catches cases where the model might work well on lighter skin but degrade on darker skin (or vice versa). Results are reported per-type in evaluation output.

### Running Evaluation

```bash
python scripts/evaluate.py \
    --pred_dir output/predictions/ \
    --target_dir data/targets/ \
    --output eval_results.json
```

The evaluation harness computes all metrics, stratifies by Fitzpatrick type and by procedure, and writes a JSON report.

---

## Clinical Edge Cases

LandmarkDiff handles four clinical conditions that affect how deformations should be applied or how the mask should behave.

### Vitiligo

Vitiligo causes depigmented patches on the skin that should be preserved, not blended over. LandmarkDiff detects vitiligo patches using LAB luminance thresholding (high L, low saturation), filters by minimum area (200 px squared), and reduces mask intensity over detected patches by a preservation factor of 0.3. This means the surgical region is still modified, but depigmented areas are largely left alone.

### Bell's Palsy

Bell's palsy causes unilateral facial paralysis. Deforming the paralyzed side produces unrealistic results because the tissue doesn't respond to surgery the same way. LandmarkDiff takes the affected side (left or right) as input and disables all deformation handles on that side. The bilateral landmark groups (eye, eyebrow, mouth corner, jawline) for the affected side are excluded from manipulation.

### Keloid-Prone Skin

Keloid-prone patients develop raised scars at incision sites. LandmarkDiff identifies keloid-prone regions (specified by anatomical zone, e.g., "jawline", "nose"), creates exclusion masks with margins, and reduces mask intensity by a factor of 0.5 with additional Gaussian blur (sigma 10.0) for softer transitions. This prevents sharp compositing boundaries that would suggest incision lines.

### Ehlers-Danlos Syndrome

Ehlers-Danlos causes tissue hypermobility - the skin stretches more than typical. LandmarkDiff multiplies the Gaussian RBF influence radius by 1.5 for Ehlers-Danlos patients, producing wider, more gradual deformations that reflect how hypermobile tissue actually responds to surgical manipulation.

### Using Clinical Flags

```python
from landmarkdiff.clinical import ClinicalFlags

flags = ClinicalFlags(
    vitiligo=True,
    bells_palsy=True,
    bells_palsy_side="left",
    keloid_prone=True,
    keloid_regions=["jawline", "nose"],
    ehlers_danlos=False,
)

result = pipeline.generate(
    image,
    procedure="rhinoplasty",
    intensity=60,
    clinical_flags=flags,
)
```

In the Gradio demo, these are checkboxes and dropdowns in Tab 1.

---

## Post-Processing Pipeline

The raw diffusion output needs refinement before it looks right. The post-processing pipeline runs six steps:

### 1. CodeFormer Face Restoration
Neural face restoration that fixes small artifacts, enhances detail, and sharpens facial features. Uses a fidelity weight of 0.7 (range 0.0 to 1.0) to balance quality enhancement against faithfulness to the diffusion output. Falls back to GFPGAN if CodeFormer is unavailable.

### 2. Real-ESRGAN Background Enhancement
Super-resolution applied only to non-face regions (background, hair, clothing). Prevents the background from looking noticeably lower quality than the restored face.

### 3. Skin Tone Matching
CDF histogram matching in LAB color space transfers the input photo's skin tone to the generated output. LAB matching is more robust than RGB for this because it separates luminance from color, preventing brightness shifts.

### 4. Frequency-Aware Sharpening
Unsharp masking applied to the L channel only (luminance) with a default strength of 0.25. Sharpening only luminance avoids the color fringing artifacts you get from sharpening RGB channels directly.

### 5. Laplacian Pyramid Blending
The compositing step - blends the generated face into the original photo. Uses a 6-level Laplacian pyramid where low-frequency levels blend smoothly (lighting and color continuity) while high-frequency levels transition sharply (texture and pore detail). This prevents the color halos and "pasted on" look that simple alpha blending produces.

### 6. ArcFace Identity Verification
Final sanity check. Extracts ArcFace embeddings from the input and output, computes cosine similarity, and flags if the score drops below 0.6. This catches cases where the diffusion model drifted too far from the patient's appearance.

---

## Project Structure

```
landmarkdiff/                   # Core library
    landmarks.py                #   MediaPipe 478-point face mesh extraction
                                #   FaceLandmarks dataclass, extract_landmarks(),
                                #   render_landmark_image(), LANDMARK_REGIONS dict
    conditioning.py             #   ControlNet conditioning generation
                                #   Tessellation wireframe (2556 edges), adaptive
                                #   Canny edge detection, generate_conditioning()
    manipulation.py             #   Gaussian RBF landmark deformation
                                #   DeformationHandle, PROCEDURE_LANDMARKS,
                                #   apply_procedure_preset(), clinical modifiers
    masking.py                  #   Feathered surgical mask generation
                                #   Convex hull + dilation + Gaussian feather +
                                #   Perlin boundary noise, clinical adjustments
    inference.py                #   Full pipeline (4 modes: tps/img2img/controlnet/
                                #   controlnet_ip), LandmarkDiffPipeline class,
                                #   face view estimation, procedure-specific prompts
    losses.py                   #   Combined loss (diffusion + landmark + identity
                                #   + perceptual), phase A/B control, procedure-
                                #   dependent identity cropping
    evaluation.py               #   Metrics (FID, LPIPS, SSIM, NME, Identity Sim),
                                #   Fitzpatrick ITA classification, per-type and
                                #   per-procedure stratification
    clinical.py                 #   Clinical edge cases: ClinicalFlags dataclass,
                                #   vitiligo patch detection, Bell's palsy side
                                #   exclusion, keloid mask adjustment, Ehlers-Danlos
    postprocess.py              #   Neural + classical post-processing: CodeFormer,
                                #   GFPGAN, Real-ESRGAN, LAB histogram matching,
                                #   Laplacian pyramid blend, ArcFace verification
    synthetic/
        pair_generator.py       #   Training pair generation pipeline
        tps_warp.py             #   Thin-plate spline warping with rigid regions
                                #   (teeth, sclera), smart control point subsampling
                                #   (max 80 from 478), batched evaluation
        augmentation.py         #   Clinical photography augmentations

scripts/                        # CLI tools
    app.py                      #   Gradio web demo (5 tabs)
    run_inference.py            #   Single image inference
    train_controlnet.py         #   ControlNet fine-tuning
    evaluate.py                 #   Automated evaluation harness
    demo.py                     #   CLI demo with visualizations
    download_ffhq.py            #   FFHQ face image downloader
    generate_synthetic_data.py  #   Synthetic training pair generator
    train_slurm.sh              #   SLURM job script (single GPU)
    train_slurm_v2.sh           #   SLURM job script (multi-GPU)
    gen_synthetic_slurm.sh      #   SLURM job for data generation

examples/                       # Runnable example scripts
    basic_inference.py          #   Single image with GPU fallback to TPS
    batch_inference.py          #   Process a directory of images
    tps_only.py                 #   CPU-only TPS warp (no GPU)
    compare_procedures.py       #   Side-by-side all procedures grid
    custom_procedure.py         #   Define a lip augmentation procedure
    landmark_visualization.py   #   Visualize mesh with displacement arrows

benchmarks/                     # Performance benchmarks
    benchmark_inference.py      #   Inference speed across hardware
    benchmark_landmarks.py      #   Landmark extraction throughput
    benchmark_training.py       #   Training steps/hour

configs/                        # Training configuration
    training.yaml               #   Default hyperparameters, loss weights, safeguards

paper/                          # MICCAI 2026 manuscript (Springer LNCS)
docs/                           # Documentation
    tutorials/                  #   quickstart, custom_procedures, training,
                                #   evaluation, deployment
    api/                        #   Per-module API reference (landmarks,
                                #   manipulation, conditioning, inference,
                                #   evaluation, clinical)
    GPU_TRAINING_GUIDE.md       #   HPC setup, Apptainer, SLURM

containers/                     # Apptainer/Singularity container definitions
tests/                          # Unit tests (9 test modules)
demos/                          # Curated sample output images
```

---

## Configuration

### Training (configs/training.yaml)

The training config controls all hyperparameters, loss weights, and safeguards. Key sections:

```yaml
model:
  controlnet: CrucibleAI/ControlNetMediaPipeFace
  base_model: runwayml/stable-diffusion-v1-5

training:
  learning_rate: 1.0e-5
  lr_scheduler: cosine
  warmup_steps: 500
  batch_size: 4
  gradient_accumulation_steps: 4  # effective batch = 16
  num_train_steps: 10000
  mixed_precision: bf16
  ema_decay: 0.9999

loss_weights:  # Phase B only
  diffusion: 1.0
  landmark: 0.1
  identity: 0.05
  perceptual: 0.1
```

### Inference Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `intensity` | 60 | 0 - 100 | How aggressive the deformation is (percentage) |
| `num_inference_steps` | 30 | 10 - 100 | Diffusion denoising steps (more = higher quality, slower) |
| `guidance_scale` | 7.5 | 1.0 - 20.0 | Classifier-free guidance strength |
| `controlnet_conditioning_scale` | 1.0 | 0.0 - 1.2 | How strongly the wireframe controls generation. Max 1.2 to avoid saturation |
| `strength` | 0.75 | 0.0 - 1.0 | img2img denoising strength |
| `seed` | None | any int | For reproducible results |

---

## Benchmarks

### Inference Speed

| Hardware | Mode | Time per image |
|----------|------|----------------|
| A100 80GB | ControlNet (30 steps) | ~3 sec |
| A100 40GB | ControlNet (30 steps) | ~4 sec |
| RTX 4090 | ControlNet (30 steps) | ~5 sec |
| RTX 3090 | ControlNet (30 steps) | ~7 sec |
| T4 16GB | ControlNet (30 steps) | ~15 sec |
| M3 Pro (MPS) | ControlNet (30 steps) | ~45 sec |
| Any CPU | TPS only | ~0.5 sec |

### Training Throughput

| Hardware | Batch size | Grad accum | Effective batch | Steps/hour |
|----------|-----------|------------|-----------------|------------|
| A100 80GB | 4 | 4 | 16 | ~600 |
| A100 40GB | 2 | 8 | 16 | ~400 |
| RTX 4090 | 2 | 8 | 16 | ~350 |
| RTX 3090 | 1 | 16 | 16 | ~200 |

### VRAM Usage

| Component | VRAM |
|-----------|------|
| SD 1.5 (FP16) | ~2.5 GB |
| ControlNet (FP16) | ~1.5 GB |
| VAE (FP32) | ~0.5 GB |
| CodeFormer | ~0.4 GB |
| ArcFace | ~0.3 GB |
| **Total inference** | **~5.2 GB** |
| **Total training** | **~25 GB** |

Run benchmarks yourself:

```bash
python benchmarks/benchmark_inference.py --device cuda --num_images 100
python benchmarks/benchmark_landmarks.py --num_images 1000
python benchmarks/benchmark_training.py --device cuda --num_steps 100
```

---

## Model Zoo

See [MODEL_ZOO.md](../MODEL_ZOO.md) for the full list of required and optional models.

**Base models (auto-downloaded on first run):**
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) - ~4 GB
- [CrucibleAI/ControlNetMediaPipeFace](https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace) - ~1.4 GB
- MediaPipe Face Mesh - ~5 MB

**Post-processing models (optional, auto-downloaded):**
- CodeFormer - ~400 MB
- GFPGAN v1.4 - ~350 MB
- Real-ESRGAN x4 - ~64 MB
- ArcFace (InsightFace buffalo_l) - ~250 MB

---

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA (or MPS on Apple Silicon)
- ~6 GB VRAM for inference (SD 1.5 + ControlNet)
- ~25 GB VRAM for training (A100 40GB minimum, 80GB recommended)
- MediaPipe 0.10.9+
- diffusers 0.27.0+, transformers 4.38.0+

Full dependency list in [pyproject.toml](pyproject.toml).

---

## Docker

```bash
# CPU-only demo (TPS mode, no GPU required)
docker build -t landmarkdiff:cpu -f Dockerfile.cpu .
docker run -p 7860:7860 landmarkdiff:cpu

# GPU-accelerated demo (ControlNet inference)
docker build -t landmarkdiff:gpu -f Dockerfile.gpu .
docker run --gpus all -p 7860:7860 landmarkdiff:gpu
```

Or with Docker Compose:

```bash
docker compose up app       # CPU demo on :7860
docker compose up gpu       # GPU demo on :7861
docker compose --profile training run train  # training (GPU)
```

GPU passthrough requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). See [Docker GPU Setup](docker-gpu.md) for prerequisites, VRAM requirements by GPU tier, and troubleshooting.

For HPC environments using Apptainer/Singularity, see [containers/](containers/).

---

## Make Targets

```bash
make help            # show all commands
make install         # install (inference only)
make install-dev     # install with dev tools
make install-train   # install with training deps
make install-app     # install with Gradio
make install-all     # install everything
make test            # run full test suite
make test-fast       # run tests excluding slow ones
make lint            # run ruff linter
make format          # auto-format code
make type-check      # run mypy
make check           # lint + format + type-check
make demo            # launch Gradio demo
make inference       # run single inference
make train           # train ControlNet
make evaluate        # run evaluation
make docker          # build Docker image
make paper           # build MICCAI paper PDF
make clean           # remove build artifacts
```

---

## Roadmap

### Current (v0.1 - Spring 2026)
- [x] Core pipeline: landmark extraction, RBF deformation, ControlNet conditioning, mask compositing
- [x] 4 procedure presets (rhinoplasty, blepharoplasty, rhytidectomy, orthognathic)
- [x] Synthetic training pair generation via TPS warps
- [x] Clinical edge case handling (vitiligo, Bell's palsy, keloid, Ehlers-Danlos)
- [x] Neural post-processing (CodeFormer, Real-ESRGAN, ArcFace identity verification)
- [x] Gradio demo with multi-angle capture
- [x] Fitzpatrick-stratified evaluation protocol
- [x] Docker and Apptainer container support
- [ ] ControlNet fine-tuning on 50K+ synthetic pairs (in progress)
- [ ] Populate results tables in paper

### Next (v0.2 - Summer 2026)
- [ ] FLUX.1-dev backbone upgrade (higher quality generation at 1024x1024)
- [ ] IP-Adapter FaceID for stronger identity preservation
- [ ] Additional procedure presets (mentoplasty, brow lift, otoplasty)
- [ ] Clinical validation with board-certified plastic surgeons
- [ ] Hugging Face Spaces interactive demo
- [ ] arXiv preprint (target: April 2026)

### Future (v1.0)
- [ ] FLAME 3D morphable model integration for depth-aware deformation
- [ ] Multi-view consistency loss across frontal/profile predictions
- [ ] Physics-informed tissue simulation (FEM for soft tissue response)
- [ ] React Native mobile capture app with standardized clinical photo acquisition
- [ ] Cloud deployment with Triton inference server

### Publication Targets
- MICCAI 2026 workshop paper (July 2026 submission)
- RSNA 2026 abstract (May 2026)
- Full conference paper (CVPR/NeurIPS 2027)

---

## Citation

```bibtex
@inproceedings{landmarkdiff2026,
  title={LandmarkDiff: Anatomically-Conditioned Latent Diffusion for
         Photorealistic Facial Surgery Outcome Prediction},
  booktitle={Medical Image Computing and Computer Assisted Intervention
             -- MICCAI},
  year={2026},
  publisher={Springer}
}
```

Machine-readable citation metadata is also available in [CITATION.cff](CITATION.cff).

---

## Contributors

We track all contributions and contributors will be acknowledged in the MICCAI 2026 paper. Significant contributions earn co-authorship.

| Contribution Level | Recognition |
|---|---|
| Bug fix or typo | Acknowledged in README |
| New procedure preset | Acknowledged in paper and README |
| Feature module (new loss, metric, clinical handler) | Co-author on paper |
| Clinical validation data | Co-author on paper |
| Sustained multi-feature contributions | Co-author on paper |

### Current Contributors

| GitHub Handle | Contribution |
|---|---|
| [@dreamlessx](https://github.com/dreamlessx) | Core architecture, training pipeline, paper |

To join this list, open a PR or contribute to an issue. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full guide, including development setup, coding style, testing requirements, and how to add new procedures.

For bug reports and feature requests, use the [issue templates](https://github.com/dreamlessx/LandmarkDiff-public/issues/new/choose).

For major changes, please open an issue first to discuss the proposed approach.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [CrucibleAI](https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace) for the MediaPipe Face ControlNet
- [MediaPipe](https://google.github.io/mediapipe/) for the 478-point face mesh
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) for the diffusion backbone
- [CodeFormer](https://github.com/sczhou/CodeFormer) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for face restoration
- [InsightFace](https://github.com/deepinsight/insightface) for ArcFace identity verification
- [FFHQ](https://github.com/NVlabs/ffhq-dataset) for training data

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started
install
tutorials/quickstart
faq
```

```{toctree}
:maxdepth: 2
:caption: Procedures

procedures/rhinoplasty
procedures/blepharoplasty
procedures/rhytidectomy
procedures/orthognathic
procedures/brow_lift
procedures/mentoplasty
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/custom_procedures
tutorials/training
tutorials/evaluation
tutorials/deployment
docker-gpu
GPU_TRAINING_GUIDE
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/landmarks
api/manipulation
api/conditioning
api/inference
api/evaluation
api/clinical
```

```{toctree}
:maxdepth: 1
:caption: Project

benchmarks
changelog
```
