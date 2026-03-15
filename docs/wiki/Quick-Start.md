# Quick Start

This guide walks through using LandmarkDiff in all four inference modes, from instant CPU-based TPS warps to full GPU-accelerated diffusion.

## Prerequisites

```bash
pip install -e .
```

For GPU modes (img2img, controlnet, controlnet_ip), you also need a CUDA-capable GPU with at least 6 GB VRAM. See the [Installation](Installation) page for details.

---

## Mode 1: TPS (Thin Plate Spline): CPU, Instant

The simplest mode. No GPU, no model downloads, no diffusion. Pure geometric warping.

```python
import cv2
from landmarkdiff.inference import LandmarkDiffPipeline

pipe = LandmarkDiffPipeline(mode="tps")
pipe.load()  # no-op for TPS mode

image = cv2.imread("face.jpg")
result = pipe.generate(
    image,
    procedure="rhinoplasty",
    intensity=60,
)

cv2.imwrite("prediction_tps.png", result["output"])
```

**When to use:** Quick previews, interactive demos, batch processing without GPU, prototyping new procedures.

**Speed:** ~50ms per image on any modern CPU.

---

## Mode 2: img2img: GPU, ~5 seconds

Feeds the TPS-warped image into Stable Diffusion 1.5 img2img for texture refinement. The diffusion model only modifies the surgical region (via mask compositing).

```python
pipe = LandmarkDiffPipeline(mode="img2img")
pipe.load()  # downloads SD1.5 (~5 GB)

result = pipe.generate(
    image,
    procedure="blepharoplasty",
    intensity=50,
    num_inference_steps=30,
    guidance_scale=9.0,
    strength=0.5,       # denoising strength (0=no change, 1=full denoise)
    seed=42,
)

cv2.imwrite("prediction_img2img.png", result["output"])
```

**When to use:** Better texture quality than TPS alone, moderate VRAM budget.

**VRAM:** ~4 GB with model_cpu_offload (enabled by default).

---

## Mode 3: ControlNet: GPU, ~8 seconds

Renders the deformed face mesh as a wireframe and uses CrucibleAI/ControlNetMediaPipeFace to generate the face from conditioning. Best photorealistic quality.

```python
pipe = LandmarkDiffPipeline(mode="controlnet")
pipe.load()  # downloads SD1.5 + ControlNet (~7 GB)

result = pipe.generate(
    image,
    procedure="rhytidectomy",
    intensity=65,
    num_inference_steps=30,
    guidance_scale=9.0,
    controlnet_conditioning_scale=0.9,
    seed=42,
)

cv2.imwrite("prediction_controlnet.png", result["output"])
```

**When to use:** Highest quality output, clinical demonstrations, publication figures.

**VRAM:** ~6 GB with model_cpu_offload.

---

## Mode 4: ControlNet + IP-Adapter: GPU, ~10 seconds

Same as ControlNet, with the addition of h94/IP-Adapter-FaceID for identity-preserving generation. Conditions the diffusion model on an ArcFace embedding of the input face.

```python
pipe = LandmarkDiffPipeline(mode="controlnet_ip", ip_adapter_scale=0.6)
pipe.load()  # downloads SD1.5 + ControlNet + IP-Adapter (~9 GB)

result = pipe.generate(
    image,
    procedure="orthognathic",
    intensity=55,
    num_inference_steps=30,
    seed=42,
)

cv2.imwrite("prediction_controlnet_ip.png", result["output"])
print(f"IP-Adapter active: {result['ip_adapter_active']}")
```

**When to use:** When identity preservation is critical, patient-facing demonstrations.

**VRAM:** ~8 GB with model_cpu_offload.

---

## Command Line Interface

All modes are accessible from the CLI:

```bash
# TPS (instant, CPU)
python -m landmarkdiff infer face.jpg --procedure rhinoplasty --intensity 60 --mode tps

# ControlNet (GPU)
python -m landmarkdiff infer face.jpg --procedure blepharoplasty --intensity 50 --mode controlnet --steps 30 --seed 42

# ControlNet + IP-Adapter
python -m landmarkdiff infer face.jpg --procedure rhytidectomy --intensity 65 --mode controlnet_ip

# Batch processing with img2img
python scripts/batch_inference.py --input-dir faces/ --output-dir results/ \
    --procedure rhinoplasty --mode img2img
```

### Visualize Landmarks

```bash
python -m landmarkdiff landmarks face.jpg --output landmarks.png
```

### Launch Gradio Demo

```bash
pip install -e ".[app]"
python -m landmarkdiff demo
# Opens http://localhost:7860
```

---

## Working with Results

The `generate()` method returns a dictionary with everything you might need:

```python
result = pipe.generate(image, procedure="rhinoplasty", intensity=60)

# Final output
output = result["output"]              # (512, 512, 3) BGR uint8

# Intermediate outputs
tps_warp = result["output_tps"]        # TPS warp result (always computed)
raw = result["output_raw"]             # raw diffusion output (before compositing)
conditioning = result["conditioning"]  # face mesh wireframe sent to ControlNet
mask = result["mask"]                  # surgical mask (float32, 0-1)

# Landmarks
original_lm = result["landmarks_original"]       # FaceLandmarks before deformation
manipulated_lm = result["landmarks_manipulated"]  # FaceLandmarks after deformation

# Metadata
print(result["procedure"])           # "rhinoplasty"
print(result["intensity"])           # 60.0
print(result["mode"])                # "tps", "img2img", etc.
print(result["view_info"])           # face orientation (yaw, pitch, view class)
print(result["manipulation_mode"])   # "preset" or "displacement_model"

# Identity check (if postprocessing ran)
if result["identity_check"]:
    print(f"Identity similarity: {result['identity_check']['similarity']:.3f}")
```

---

## Step-by-Step Pipeline Walkthrough

For finer control, you can run each pipeline stage manually:

```python
import cv2
import numpy as np
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.synthetic.tps_warp import warp_image_tps

# 1. Load and resize
image = cv2.imread("face.jpg")
image = cv2.resize(image, (512, 512))

# 2. Extract landmarks
face = extract_landmarks(image)
assert face is not None, "No face detected"
print(f"Detected {face.landmarks.shape[0]} landmarks, confidence={face.confidence}")

# 3. Apply surgical deformation
deformed = apply_procedure_preset(
    face,
    procedure="rhinoplasty",
    intensity=60.0,
    image_size=512,
)

# 4. Generate conditioning wireframe (for ControlNet)
wireframe = render_wireframe(deformed, 512, 512)
mesh_image = render_landmark_image(deformed, 512, 512)

# 5. Generate surgical mask
mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)

# 6. TPS warp
warped = warp_image_tps(image, face.pixel_coords, deformed.pixel_coords)

# 7. Save intermediate outputs
cv2.imwrite("wireframe.png", wireframe)
cv2.imwrite("mesh.png", mesh_image)
cv2.imwrite("mask.png", (mask * 255).astype(np.uint8))
cv2.imwrite("warped.png", warped)
```

---

## All Six Procedures

```python
procedures = [
    "rhinoplasty",      # nose reshaping
    "blepharoplasty",   # eyelid surgery
    "rhytidectomy",     # facelift
    "orthognathic",     # jaw repositioning
    "brow_lift",        # brow elevation
    "mentoplasty",      # chin surgery
]

for proc in procedures:
    result = pipe.generate(image, procedure=proc, intensity=60)
    cv2.imwrite(f"output_{proc}.png", result["output"])
```

---

## Data-Driven Displacement Model

If you have a fitted `DisplacementModel` (from real before/after surgery pairs), the pipeline can use learned displacement vectors instead of hand-tuned presets:

```python
pipe = LandmarkDiffPipeline(
    mode="controlnet",
    displacement_model_path="data/displacement_model.npz",
)
pipe.load()

result = pipe.generate(image, procedure="rhinoplasty", intensity=50)
print(f"Manipulation mode: {result['manipulation_mode']}")
# "displacement_model" if the model has data for rhinoplasty, "preset" otherwise
```

---

## Clinical Flags

For patients with specific clinical conditions:

```python
from landmarkdiff.clinical import ClinicalFlags

flags = ClinicalFlags(
    vitiligo=True,          # preserve depigmented patches
    bells_palsy=True,       # skip deformation on paralyzed side
    bells_palsy_side="left",
    keloid_prone=True,      # soften mask transitions
    keloid_regions=["jawline"],
    ehlers_danlos=True,     # wider deformation radii
)

result = pipe.generate(image, procedure="rhytidectomy", intensity=50, clinical_flags=flags)
```

See the [Clinical Flags](Clinical-Flags) page for detailed behavior.

---

## Interactive Notebook

For a guided walkthrough with inline visualizations, see the [quickstart notebook](https://github.com/dreamlessx/LandmarkDiff-public/blob/main/notebooks/quickstart.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dreamlessx/LandmarkDiff-public/blob/main/notebooks/quickstart.ipynb)

---

## Next Steps

- [Procedures](Procedures): detailed guide for each surgical procedure
- [API Reference](API-Reference): full class and function documentation
- [Architecture](Architecture): how the pipeline works internally
- [Training](Training): how to train your own ControlNet
