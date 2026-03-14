"""Generate a model card (Markdown) for LandmarkDiff.

Creates a comprehensive model card following the framework of Mitchell et al.
(2019) "Model Cards for Model Reporting". This is increasingly expected at
top venues for responsible AI and helps document intended use, limitations,
and evaluation methodology.

Usage:
    python scripts/generate_model_card.py --output paper/MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_eval_results(paper_dir: Path) -> dict | None:
    """Try to load evaluation results from various paths."""
    for name in [
        "eval_results_multiseed.json",
        "eval_results_aggregated.json",
        "phaseA_eval_results.json",
    ]:
        path = paper_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def load_baseline_results(paper_dir: Path) -> dict | None:
    for name in ["baseline_results.json"]:
        path = paper_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def generate_model_card(output_path: str) -> None:
    """Generate the model card Markdown file."""
    paper_dir = ROOT / "paper"
    eval_data = load_eval_results(paper_dir)
    load_baseline_results(paper_dir)

    # Build metrics table if data available
    metrics_table = ""
    if eval_data:
        metrics_table = "\n### Evaluation Results\n\n"
        metrics_table += "| Metric | Value | Description |\n"
        metrics_table += "|--------|-------|-------------|\n"

        if "metrics" in eval_data:
            m = eval_data["metrics"]
            for key, desc in [
                ("ssim", "Structural similarity (higher = better)"),
                ("lpips", "Perceptual distance (lower = better)"),
                ("nme", "Normalized landmark error (lower = better)"),
                ("identity_sim", "ArcFace cosine similarity (higher = better)"),
                ("fid", "Frechet Inception Distance (lower = better)"),
            ]:
                if key in m:
                    val = m[key]
                    if isinstance(val, dict):
                        metrics_table += (
                            f"| {key.upper()} | {val.get('mean', val):.4f} | {desc} |\n"
                        )
                    else:
                        metrics_table += f"| {key.upper()} | {val:.4f} | {desc} |\n"

        if "per_procedure" in eval_data:
            metrics_table += "\n### Per-Procedure Performance\n\n"
            metrics_table += "| Procedure | SSIM | LPIPS | NME | ArcFace | n |\n"
            metrics_table += "|-----------|------|-------|-----|---------|---|\n"
            for proc in sorted(eval_data["per_procedure"]):
                pp = eval_data["per_procedure"][proc]

                def _v(d, k):
                    v = d.get(k, 0)
                    return v.get("mean", v) if isinstance(v, dict) else v

                metrics_table += (
                    f"| {proc} | {_v(pp, 'ssim'):.3f} | {_v(pp, 'lpips'):.3f} | "
                    f"{_v(pp, 'nme'):.3f} | {_v(pp, 'identity_sim'):.3f} | "
                    f"{pp.get('count', pp.get('n', '?'))} |\n"
                )

    card = f"""---
language: en
tags:
  - diffusion
  - controlnet
  - facial-surgery
  - medical-imaging
  - image-generation
license: cc-by-nc-4.0
pipeline_tag: image-to-image
---

# LandmarkDiff: Anatomically-Conditioned Latent Diffusion for Facial Surgery Outcome Prediction

## Model Description

**LandmarkDiff** is a latent diffusion model that generates photorealistic
predictions of facial surgery outcomes from a single 2D photograph. It uses
MediaPipe FaceMesh landmarks with procedure-specific Gaussian RBF deformation
fields to condition a ControlNet-augmented Stable Diffusion 1.5 backbone.

- **Architecture**: SD1.5 + CrucibleAI ControlNet (MediaPipe Face)
- **Input**: Single clinical photograph (any resolution, resized to 512x512)
- **Output**: Predicted post-operative appearance (512x512)
- **Procedures**: Rhinoplasty, Blepharoplasty, Rhytidectomy, Orthognathic surgery
- **Training data**: Synthetic pairs (CelebA-HQ + FFHQ via TPS warping) + clinical pairs (HDA database)

## Intended Use

### Primary Use Case
Pre-operative consultation in outpatient cosmetic clinics: showing patients
what they might look like after a specific facial procedure, using only a
smartphone photograph as input. Replaces expensive 3D imaging (CT/CBCT) and
crude pixel-morphing tools.

### Out-of-Scope Uses
- **NOT for diagnosis**: This is a visualization tool, not a diagnostic system
- **NOT for unsupervised patient use**: Should always be presented with surgeon supervision
- **NOT for identity verification**: Predictions are approximations, not exact replicas
- **NOT for non-facial procedures**: Only trained on the four specified facial procedures
- **NOT for profile views**: Trained on frontal photographs only; depth-dependent effects
  (e.g., dorsal hump reduction in profile) are not captured

## Training

### Phase A: Synthetic Pre-training
- **Data**: 10,000 synthetic before/after pairs from CelebA-HQ and FFHQ via TPS warping
- **Steps**: 50,000
- **Loss**: Standard diffusion MSE loss
- **Purpose**: Learn face synthesis conditioned on landmark meshes

### Phase B: Clinical Fine-tuning
- **Data**: ~466 real before/after pairs from HDA Plastic Surgery Face Database + synthetic
- **Steps**: 25,000
- **Loss**: Multi-term: diffusion + LPIPS perceptual + ArcFace identity + curriculum weighting
- **Purpose**: Bridge synthetic-to-clinical domain gap

### Hardware
- Single NVIDIA A6000 (48 GB VRAM)
- BF16 mixed precision
- AdamW optimizer, cosine LR schedule

{metrics_table}

## Limitations and Bias

### Known Limitations
1. **2D only**: Cannot capture depth-dependent effects visible in profile views
2. **Healed outcome**: Predicts final result, not post-operative swelling/bruising/scarring
3. **Identity preservation**: Phase A training may show identity collapse for procedures
   involving large geometric changes (e.g., rhytidectomy). Phase B addresses this.
4. **Limited procedures**: Only four facial procedures supported

### Bias and Fairness
- Training data (CelebA-HQ, FFHQ) under-represents Fitzpatrick Types V and VI
- Evaluation is stratified by Fitzpatrick skin type (ITA-based classification)
- LAB color matching may produce visible artifacts on very dark skin tones
- Clinical validation is limited to the HDA database demographics

### Ethical Considerations
- Predictions should **never** be presented as guarantees of surgical outcomes
- Patients must be informed that the visualization is an AI approximation
- The system should not be used to pressure patients into surgery
- Unrealistic expectations management remains the surgeon's responsibility

## Citation

```bibtex
@inproceedings{{landmarkdiff2026,
  title={{{{LandmarkDiff}}}}: Anatomically-Conditioned Latent Diffusion for
         Photorealistic Facial Surgery Outcome Prediction}},
  author={{Anonymous}},
  booktitle={{Medical Image Computing and Computer Assisted Intervention -- MICCAI}},
  year={{2026}},
  publisher={{Springer}}
}}
```

## Model Card Contact

For questions about the model, please open an issue on the
[GitHub repository](https://github.com/dreamlessx/LandmarkDiff).

---
*Model card generated on {datetime.now().strftime("%Y-%m-%d")}*
"""

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card)
    print(f"Model card saved to {out_path}")
    print(f"Length: {len(card)} chars, {card.count(chr(10))} lines")


def main():
    parser = argparse.ArgumentParser(description="Generate model card")
    parser.add_argument("--output", type=str, default="paper/MODEL_CARD.md")
    args = parser.parse_args()
    generate_model_card(args.output)


if __name__ == "__main__":
    main()
