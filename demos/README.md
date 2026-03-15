# Demo Outputs

## Pipeline Visualization

Step-by-step visualization of the LandmarkDiff pipeline stages.

| File | Description |
|------|-------------|
| `demo_pipeline_0.png` | Pipeline stages -- Subject 1 |
| `demo_pipeline_1.png` | Pipeline stages -- Subject 2 |

Each image shows: **Input | Original Mesh | Manipulated Mesh | Surgical Mask | Result**

## Procedure Comparison (TPS Mode)

All 6 supported procedures applied at intensity 60 using the TPS (CPU-only) pipeline.

| File | Description |
|------|-------------|
| `procedure_comparison.png` | Primary grid: original input + all 6 procedure results |
| `procedure_comparison_all_subjects.png` | All 3 demo faces, before/after for each procedure |
| `procedure_comparison_0.png` | Before/after grid -- Subject 1 |
| `procedure_comparison_1.png` | Before/after grid -- Subject 2 |
| `procedure_comparison_2.png` | Before/after grid -- Subject 3 |

## Abstract Diagrams

| File | Description |
|------|-------------|
| `pipeline_abstract.png` | Five-stage pipeline flowchart (no faces) |
| `mesh_deformation.png` | Original vs deformed mesh with displacement vectors |

## Photorealistic Results

ControlNet-generated photorealistic demos will be added here once model training is complete. The pipeline is fully implemented -- we are currently training on synthetic data and will update this directory with proper results.

To generate your own demos:

```bash
# TPS mode (CPU, geometric only)
python examples/tps_only.py /path/to/face.jpg --procedure rhinoplasty --intensity 60

# ControlNet mode (GPU, photorealistic)
python scripts/run_inference.py /path/to/face.jpg --procedure rhinoplasty --intensity 60 --mode controlnet
```
