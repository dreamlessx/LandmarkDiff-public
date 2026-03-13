# Quick Start

Run your first surgical prediction in 5 minutes.

## 1. Install

```bash
pip install -e .
```

## 2. Prepare an input image

Use any front-facing photo. The face should be clearly visible with good lighting.

## 3. Run a prediction

```bash
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 0.6
```

This will:
1. Detect 478 facial landmarks using MediaPipe
2. Deform the nose landmarks according to rhinoplasty displacement vectors
3. Render a deformed tessellation mesh
4. Generate a photorealistic prediction via ControlNet + Stable Diffusion
5. Composite the result with skin tone matching
6. Save the output to `output/`

## 4. Try different procedures

```bash
# Eyelid surgery
python scripts/run_inference.py photo.jpg --procedure blepharoplasty --intensity 0.5

# Facelift
python scripts/run_inference.py photo.jpg --procedure rhytidectomy --intensity 0.7

# Jaw surgery
python scripts/run_inference.py photo.jpg --procedure orthognathic --intensity 0.6
```

## 5. Adjust intensity

The `--intensity` flag controls how dramatic the change is (0.0 = no change, 1.0 = maximum).

```bash
# Subtle change
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 0.3

# Moderate
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 0.6

# Dramatic
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --intensity 0.9
```

## 6. Launch the interactive demo

```bash
python scripts/app.py
```

Open http://localhost:7860 in your browser. The demo provides:
- Drag-and-drop image upload
- Procedure selection and intensity slider
- Side-by-side before/after comparison
- Multi-angle capture for comprehensive predictions

## 7. Use TPS mode (no GPU required)

If you don't have a GPU, use TPS (thin-plate spline) mode for geometric-only predictions:

```bash
python scripts/run_inference.py photo.jpg --procedure rhinoplasty --mode tps
```

This runs entirely on CPU but produces less photorealistic results.

## Next Steps

- [Custom Procedures](custom_procedures.md) - Define your own surgical procedure
- [Training](training.md) - Train on your own data
- [Model Zoo](../../MODEL_ZOO.md) - Download pre-trained checkpoints
