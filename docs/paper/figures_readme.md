# Reproducing Paper Figures

All figures for the LandmarkDiff paper can be generated using `scripts/generate_paper_figures.py`. This document explains which functions produce which figures, what data is required, and where output files land.

## Quick Start

```bash
# Generate all figures at once
python scripts/generate_paper_figures.py \
    --input data/faces_all/000001.png \
    --output paper/figures/ \
    --figure all

# Generate a single figure
python scripts/generate_paper_figures.py \
    --input data/faces_all/000001.png \
    --output paper/figures/ \
    --figure architecture
```

## Figures

### Figure 1: Architecture Diagram (`fig_architecture.pdf` / `.png`)

- **Function:** `figure_architecture()`
- **Input data:** None (purely diagrammatic)
- **Dependencies:** matplotlib
- **Description:** Six-block pipeline diagram showing Input Image -> MediaPipe 478-pt Mesh -> Gaussian RBF Deformation -> ControlNet (CrucibleAI) -> SD 1.5 Diffusion -> Post-Processing, with sub-labels for post-processing stages.

### Figure 2: Procedure Grid (`fig_procedure_grid.png`)

- **Function:** `figure_procedure_grid()`
- **Input data:** A single face image (512x512 or any resolution; resized internally)
- **Description:** 6-procedure x 3-intensity comparison grid. Rows are rhinoplasty, blepharoplasty, rhytidectomy, orthognathic, brow lift, mentoplasty. Columns are 33%, 66%, 100% intensity. Each cell shows the TPS-warped result composited via the surgical mask.

### Figure 3: Deformation Overlay (`fig_deformation_<procedure>.png`)

- **Function:** `figure_deformation_overlay()`
- **Input data:** A single face image
- **Description:** Three-panel visualization for each procedure: original landmarks (green dots), displacement arrows (green->red), and warped result with deformed landmarks (red dots). Generated for all 6 procedures when `--figure all` is used.

### Figure 4: Post-Processing Comparison (`fig_postprocess.png`)

- **Function:** `figure_postprocess_comparison()`
- **Input data:** A single face image
- **Description:** Six-panel strip showing the output at each post-processing stage: Original, TPS Warp, + Histogram Match, + Sharpening, + Laplacian Blend, Final Output. Uses rhinoplasty at 65% intensity.

### Figure 5: Pipeline Overview (`fig_pipeline.png`)

- **Function:** `figure_pipeline()`
- **Input data:** A single face image
- **Description:** Seven-panel strip: Input, Landmarks, Wireframe, Canny, Mask, Warped, Output. Shows every intermediate representation in the pipeline for rhinoplasty at 65% intensity.

### Figure 6: Conditioning Ablation (`fig_conditioning.png`)

- **Function:** `figure_conditioning()`
- **Input data:** A single face image
- **Description:** Six-panel comparison of conditioning signals: Input, Wireframe only, Canny only, Mesh + Canny, Tessellation, Full Conditioning. Useful for the ablation study section of the paper.

## Required Data

- **Face images:** Any face photo works. The script looks for `data/faces_all/000001.png` or `data/celeba_hq_extracted/000001.png` by default. You can override with `--input <path>` or `--input_dir <directory>`.
- **No model weights needed.** All figures use TPS warping (CPU-only), so no GPU or diffusion model downloads are required.
- **matplotlib** is needed only for the architecture diagram (Figure 1). All other figures use OpenCV.

## Output Locations

By default, all figures are written to `paper/figures/`. Override with `--output <path>`.

```
paper/figures/
    fig_architecture.pdf        # Vector format for LaTeX
    fig_architecture.png        # Raster preview
    fig_procedure_grid.png
    fig_deformation_rhinoplasty.png
    fig_deformation_blepharoplasty.png
    fig_deformation_rhytidectomy.png
    fig_deformation_orthognathic.png
    fig_deformation_brow_lift.png
    fig_deformation_mentoplasty.png
    fig_postprocess.png
    fig_pipeline.png
    fig_conditioning.png
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | None | Path to a single input face image |
| `--input_dir` | None | Directory of face images (uses first `.png` found) |
| `--output` | `paper/figures` | Output directory for generated figures |
| `--max_images` | 4 | Max images to process from `--input_dir` |
| `--size` | 256 | Panel size in pixels (per cell) |
| `--dpi` | 200 | DPI for vector figures (architecture diagram) |
| `--figure` | `all` | Which figure to generate: `all`, `architecture`, `procedure_grid`, `deformation`, `postprocess`, `pipeline`, `conditioning` |

## Additional Figure Scripts

### General

- `scripts/generate_figures.py`: general-purpose figure generation
- `scripts/generate_qualitative_figure.py`: qualitative comparison grids
- `scripts/generate_paper_tables.py`: LaTeX tables for the results section
- `scripts/populate_paper_tables.py`: fills table placeholders with computed metrics
- `scripts/json_to_latex.py`: converts JSON evaluation results to LaTeX tables

### New in v0.2.2

- `scripts/build_qualitative_grid.py`: multi-procedure qualitative comparison grid suitable for the main paper figure. Accepts a directory of input faces and produces a single composited PNG with one row per face and one column per procedure.

  ```bash
  python scripts/build_qualitative_grid.py \
      --input_dir data/test_faces/ \
      --output paper/figures/fig_qualitative_grid.png \
      --procedures rhinoplasty blepharoplasty rhytidectomy orthognathic brow_lift mentoplasty \
      --intensity 65
  ```

- `scripts/generate_comparison_figure.py`: side-by-side comparison of LandmarkDiff against baseline methods (TPS-only, SD img2img). Used for Table 1 row visualization.

  ```bash
  python scripts/generate_comparison_figure.py \
      --input data/test_faces/000001.png \
      --output paper/figures/fig_comparison.png \
      --procedure rhinoplasty
  ```

- `scripts/landmark_accuracy_heatmap.py`: generates a 478-point landmark error heatmap overlaid on the canonical face mesh. Color-codes per-landmark NME from a saved evaluation JSON.

  ```bash
  python scripts/landmark_accuracy_heatmap.py \
      --results eval/landmark_errors.json \
      --output paper/figures/fig_landmark_heatmap.png \
      --procedure rhinoplasty
  ```

- `scripts/visualize_attention.py`: extracts and visualizes ControlNet cross-attention maps at each denoising step. Produces per-step attention overlays and a summary strip.

  ```bash
  python scripts/visualize_attention.py \
      --input data/test_faces/000001.png \
      --checkpoint weights/controlnet/ \
      --output paper/figures/fig_attention/ \
      --procedure rhinoplasty \
      --steps 30
  ```

- `scripts/visualize_latent_space.py`: projects latent codes from a set of generated images into 2D via UMAP or PCA. Points are colored by procedure, producing a scatter plot that shows procedure-level clustering.

  ```bash
  python scripts/visualize_latent_space.py \
      --input_dir data/test_faces/ \
      --checkpoint weights/controlnet/ \
      --output paper/figures/fig_latent_space.png \
      --projection umap
  ```

- `scripts/progressive_denoising.py`: renders a horizontal strip showing the diffusion output at selected denoising steps (e.g., t=1000, 800, 600, 400, 200, 0). Useful for the supplementary denoising trajectory figure.

  ```bash
  python scripts/progressive_denoising.py \
      --input data/test_faces/000001.png \
      --checkpoint weights/controlnet/ \
      --output paper/figures/fig_denoising.png \
      --procedure rhinoplasty \
      --steps 30
  ```

- `scripts/intensity_sweep.py`: generates a horizontal strip of outputs at 0%, 20%, 40%, 60%, 80%, 100% intensity for a single face and procedure. Used in the supplementary S1 tables and figures.

  ```bash
  python scripts/intensity_sweep.py \
      --input data/test_faces/000001.png \
      --output paper/figures/fig_intensity_sweep_rhinoplasty.png \
      --procedure rhinoplasty
  ```
