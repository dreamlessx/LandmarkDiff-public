# Demo Outputs

Sample outputs from the LandmarkDiff pipeline showing landmark-guided facial surgery predictions.

## Procedure Comparisons

Side-by-side comparison of all four supported procedures applied to the same input face at 60% intensity.

| File | Description |
|------|-------------|
| `demo_0.png` | All procedures - Subject 1 |
| `demo_2.png` | All procedures - Subject 2 |
| `demo_3.png` | All procedures - Subject 3 |

Each image shows: **Original | Rhinoplasty | Blepharoplasty | Rhytidectomy | Orthognathic**

## Pipeline Visualization

Step-by-step visualization of the LandmarkDiff pipeline stages.

| File | Description |
|------|-------------|
| `demo_pipeline_0.png` | Pipeline stages - Subject 1 |
| `demo_pipeline_1.png` | Pipeline stages - Subject 2 |

Each image shows: **Input | Original Mesh | Manipulated Mesh | Surgical Mask | Result**

## Individual Procedure Results

Per-subject, per-procedure outputs showing the original face, deformed mesh conditioning, and predicted outcome at 60% intensity.

Files follow the naming convention: `tps_{subject}_{procedure}_60.png`

Each image shows: **Original | Mesh Conditioning | Predicted Outcome**

Subjects: 0, 1, 2, 3, 5, 8 (FFHQ samples)
Procedures: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic

## Highlight Reel

`tps_highlight_reel.png` - Compact grid showing the best results across multiple subjects and procedures.

## Notes

- All demo images use FFHQ face samples at 512x512 resolution
- Results shown use TPS (thin-plate spline) warping for geometric deformation
- ControlNet-based photorealistic generation will be showcased after model training completes
- Intensity is set to 60% for a natural, clinically plausible range of change
