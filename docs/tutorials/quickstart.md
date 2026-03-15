# Quick Start

Get your first surgical prediction running in 5 minutes.

## 1. Install

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public
pip install -e .
```

This installs the core package with all inference dependencies (MediaPipe, PyTorch, diffusers, transformers).

For GPU acceleration (xformers + triton):

```bash
pip install -e ".[gpu]"
```

Verify the installation:

```bash
python -c "
import landmarkdiff
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
print(f'LandmarkDiff {landmarkdiff.__version__} installed successfully')
"
```

## 2. Prepare an Input Image

Use any front-facing photo. For best results:
- Clear, well-lit face
- Frontal pose (slight angles are fine)
- At least 256x256 resolution (images are resized to 512x512 internally)
- No sunglasses or face masks

## 3. Run Your First Prediction (TPS mode, no GPU)

TPS mode uses thin-plate spline warping and runs entirely on CPU. It is the fastest way to see LandmarkDiff in action:

```bash
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --mode tps --intensity 50
```

This will:
1. Detect 478 facial landmarks using MediaPipe
2. Deform nose landmarks according to the rhinoplasty preset
3. Apply a thin-plate spline warp to the face image
4. Save the output to `output/`

The result appears in `output/photo_rhinoplasty/before_after.png` as a side-by-side comparison.

### Python API

The same thing in three lines of Python:

```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="tps", device="cpu")
result = pipeline.generate("photo.jpg", procedure="rhinoplasty", intensity=60)
```

The `result` dictionary contains:
- `result["output"]`: the final composited image
- `result["input"]`: the original 512x512 input
- `result["conditioning"]`: the deformed wireframe mesh overlay
- `result["mask"]`: the surgical region mask

## 4. GPU Mode (Photorealistic Results)

If you have a GPU with 6+ GB VRAM, use ControlNet mode for diffusion-based photorealistic output:

```bash
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --mode controlnet --intensity 50
```

Or in Python:

```python
pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
result = pipeline.generate("photo.jpg", procedure="rhinoplasty", intensity=60)
```

The first run downloads the SD1.5 and ControlNet weights (~5 GB). Subsequent runs reuse the cached models.

## 5. Try Different Procedures

LandmarkDiff supports 6 procedures:

```bash
# Nose reshaping
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 50

# Eyelid surgery
python scripts/run_inference.py photo.jpg --procedure blepharoplasty --intensity 50

# Facelift
python scripts/run_inference.py photo.jpg --procedure rhytidectomy --intensity 60

# Jaw surgery
python scripts/run_inference.py photo.jpg --procedure orthognathic --intensity 50

# Brow lift
python scripts/run_inference.py photo.jpg --procedure brow_lift --intensity 45

# Chin reshaping
python scripts/run_inference.py photo.jpg --procedure mentoplasty --intensity 50
```

## 6. Adjust Intensity

The `--intensity` flag controls how dramatic the change is (0 = no change, 100 = maximum surgical effect):

```bash
# Subtle
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 25

# Moderate
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 50

# Aggressive
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 85
```

A good starting point for most procedures is 40-60.

## 7. Launch the Interactive Demo

The Gradio web demo provides a visual interface with drag-and-drop upload and real-time sliders:

```bash
pip install -e ".[app]"
python scripts/app.py
```

Open http://localhost:7860 in your browser. The demo provides:
- Drag-and-drop image upload
- Procedure selection dropdown
- Intensity slider
- Side-by-side before/after comparison

A hosted demo is also available at [huggingface.co/spaces/dreamlessx/LandmarkDiff](https://huggingface.co/spaces/dreamlessx/LandmarkDiff) (TPS mode, CPU).

## 8. Combine Procedures

Apply multiple procedures sequentially for comprehensive facial planning:

```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="tps", device="cpu")

# First pass: rhinoplasty
result = pipeline.generate("photo.jpg", procedure="rhinoplasty", intensity=50)

# Save intermediate, then apply chin work
from PIL import Image
import numpy as np

Image.fromarray(result["output"]).save("step1.jpg")
result2 = pipeline.generate("step1.jpg", procedure="mentoplasty", intensity=40)
```

Or at the landmark level for more precise control:

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

face = extract_landmarks(image)
step1 = apply_procedure_preset(face, "rhinoplasty", intensity=50)
step2 = apply_procedure_preset(step1, "mentoplasty", intensity=40)
```

## Next Steps

- [Custom Procedures](custom_procedures.md): define your own surgical procedure preset
- [Training](training.md): train on your own data
- [Evaluation](evaluation.md): evaluate prediction quality with metrics
- [Deployment](deployment.md): deploy as an API or Docker service
- [API Reference](../api/landmarks.md): full module documentation
- [Procedure Docs](../procedures/rhinoplasty.md): detailed anatomy and displacement info for each procedure
- [Interactive Notebook](https://github.com/dreamlessx/LandmarkDiff-public/blob/main/notebooks/quickstart.ipynb): step-by-step Jupyter notebook with visualizations
