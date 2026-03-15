# Architecture

LandmarkDiff processes a single face photograph through a multi-stage pipeline to produce a predicted post-surgical appearance. The pipeline has four major stages: landmark extraction, deformation, conditioning/generation, and post-processing.

## Pipeline Overview

```
Input Image (512x512 BGR)
    |
    v
[1] MediaPipe Face Mesh v2
    |   478 landmarks (normalized x,y,z)
    |   FaceLandmarks dataclass
    v
[2] Gaussian RBF Deformation
    |   Procedure-specific displacement vectors
    |   Per-landmark influence radii
    |   Optional: DisplacementModel (data-driven)
    v
[3] Generation (4 modes)
    |
    +---> TPS Warp (CPU, instant)
    |       Thin-plate spline geometric warp
    |
    +---> img2img (GPU, ~5s)
    |       SD1.5 img2img on TPS-warped image
    |       Mask compositing into original
    |
    +---> ControlNet (GPU, ~8s)
    |       CrucibleAI/ControlNetMediaPipeFace
    |       Face mesh wireframe as conditioning
    |       DPM++ 2M Karras scheduler
    |
    +---> ControlNet + IP-Adapter (GPU, ~10s)
    |       ControlNet with h94/IP-Adapter-FaceID
    |       Identity-preserving generation
    v
[4] Post-Processing
    |   CodeFormer face restoration
    |   Real-ESRGAN background enhancement
    |   Histogram matching (LAB space)
    |   Frequency-aware sharpening
    |   Laplacian pyramid blending
    |   ArcFace identity verification
    v
Final Output (512x512 BGR)
```

## Stage 1: Landmark Extraction

MediaPipe Face Mesh v2 detects 478 landmarks on the input face:
- 468 face surface points covering the jawline, eyes, nose, lips, eyebrows, cheeks, and forehead
- 10 iris landmarks (5 per eye)

The code tries the MediaPipe Tasks API first (>= 0.10.20), then falls back to the legacy Solutions API. Landmarks are stored as normalized (x, y, z) coordinates in the `FaceLandmarks` dataclass.

Landmarks are grouped into named anatomical regions: `jawline`, `eye_left`, `eye_right`, `eyebrow_left`, `eyebrow_right`, `nose`, `lips`, `iris_left`, `iris_right`. Each region has associated MediaPipe indices and a visualization color.

## Stage 2: Gaussian RBF Deformation

Each procedure defines:
- A set of target landmark indices (`PROCEDURE_LANDMARKS`)
- A default influence radius in pixels at 512x512 (`PROCEDURE_RADIUS`)
- Procedure-specific displacement vectors in `_get_procedure_handles()`

The deformation formula per landmark is:

```
new_position = old_position + displacement * exp(-dist^2 / (2 * radius^2))
```

Where `dist` is the Euclidean distance from the current landmark to the handle's center landmark. This creates smooth, anatomically plausible deformations that fall off with distance.

Intensity is specified on a 0-100 scale. Internally, `apply_procedure_preset()` divides by 100 to get a scale factor applied to all displacement vectors.

### Data-Driven Mode

When a `DisplacementModel` (fitted from real before/after surgery pairs) is available, the pipeline uses learned mean displacement vectors instead of hand-tuned RBF handles. The model stores per-procedure, per-landmark statistics (mean, std, min, max, median) and can generate stochastic displacement fields with controllable variation.

## Stage 3: Generation

### TPS Mode (CPU)
Pure geometric thin-plate spline warp. Maps original landmark positions to deformed positions and interpolates the full image. No diffusion model needed: results are instant but lack the photorealistic texture synthesis of diffusion modes.

### img2img Mode
Feeds the TPS-warped image into SD1.5 img2img pipeline with a procedure-specific prompt. A surgical mask (convex hull of procedure landmarks, dilated + feathered) controls the compositing region.

### ControlNet Mode
Renders the deformed face mesh as a wireframe (2556-edge tessellation matching what CrucibleAI's ControlNet was trained on) and uses it as conditioning. The full MediaPipe tessellation and contour edges are drawn: thin gray lines for tessellation, brighter white for contours. Uses DPM++ 2M Karras scheduler for photorealistic output.

### ControlNet + IP-Adapter Mode
Extends ControlNet mode with h94/IP-Adapter-FaceID to condition generation on the input face embedding. This provides stronger identity preservation. The IP-Adapter scale defaults to 0.6.

## Stage 4: Post-Processing

The post-processing pipeline applies six steps:

1. **CodeFormer**: Transformer-based face restoration using codebook lookup. Fidelity parameter (default 0.7) balances quality vs. identity preservation. Falls back to GFPGAN if unavailable.

2. **Real-ESRGAN**: Neural 4x upscaler applied only to background (non-face) regions, then downsampled back. Improves overall sharpness without interfering with the face restoration.

3. **Histogram matching**: CDF-based color matching in LAB space within the mask region. Ensures the generated face matches the original skin tone distribution, not just mean/std.

4. **Frequency-aware sharpening**: Unsharp mask in LAB luminance channel only. Recovers fine skin texture (pores, fine lines) that diffusion tends to smooth out. Avoids color fringing by working in luminance only.

5. **Laplacian pyramid blending**: Multi-band blending at 6 pyramid levels. Low frequencies blend smoothly (no color seams), high frequencies transition sharply (preserving hair/texture boundaries). Replaces the visible halos from simple alpha blending.

6. **ArcFace identity verification**: Computes cosine similarity between ArcFace embeddings of the original and output. Flags identity drift if similarity drops below 0.6 (configurable threshold).

## Mask Generation

Surgical masks are generated deterministically from procedure-specific landmark subsets:

1. Compute convex hull of the procedure's landmark indices
2. Apply morphological dilation (15-40 px depending on procedure)
3. Add boundary noise (2-4 px Perlin-style) to prevent clean-edge seams
4. Gaussian feathering (sigma 10-20) for smooth alpha gradient

Clinical flags can modify the mask: vitiligo patches get reduced mask intensity to preserve depigmented regions, and keloid-prone areas get softened transitions.

## Device Support

| Device | Modes | Precision | Notes |
|--------|-------|-----------|-------|
| CUDA | All 4 | fp16 | model_cpu_offload enabled by default |
| MPS (Apple Silicon) | All 4 | fp32 | attention_slicing enabled |
| CPU | tps only | fp32 | Diffusion modes too slow on CPU |
