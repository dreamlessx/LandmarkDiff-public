# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-03-15

### Added
- `NasalMorphometry` module for computing nasal ratios from Varghaei et al. (2025)
- `FacialSymmetry` bilateral symmetry scoring via KDTree-based matching
- `FaceLandmarks.pixel_coords_at()` for getting coordinates at any resolution
- `FaceLandmarks.rescale()` for creating resolution-adjusted copies
- `collect_instagram_data.py` script for curating public-domain training pairs
- Competitive landscape table in README (9-method comparison)

### Fixed
- `DisplacementModel.load()` now raises `ValueError` on corrupted/empty `.npz` files instead of returning zero-initialized model
- `get_face_embedding()` guards against near-zero ArcFace embeddings from occluded faces, preventing NaN similarity scores
- CUDA OOM during inference now raises an informative `RuntimeError` instead of hanging
- Remaining `print()` calls migrated to `logging` module

## [0.2.2] - 2026-03-14

### Added
- LCM-LoRA fast inference mode in `inference.py` (latent consistency distillation, 4-step generation)
- Per-procedure validation pass with anatomy-aware rejection logic
- 26 new utility scripts synced from upstream (evaluation, visualization, paper figure generation)
- GitHub wiki expanded to 15 pages (procedure guides, training walkthrough, evaluation reference)
- Real demo faces in HF Space replacing placeholder images
- `build_qualitative_grid.py`: multi-procedure qualitative comparison grid
- `generate_comparison_figure.py`: side-by-side method comparison figure
- `landmark_accuracy_heatmap.py`: per-landmark error heatmap for paper
- `visualize_attention.py`: ControlNet cross-attention map visualization
- `visualize_latent_space.py`: UMAP/PCA projection of latent codes by procedure
- `progressive_denoising.py`: denoising trajectory strip for supplementary
- `intensity_sweep.py`: full intensity sweep grid (0-100%) for all procedures

### Changed
- bf16 is now the default training precision (was fp32)
- Loss weight tuning: landmark weight 0.1, identity weight 0.05, perceptual weight 0.1 (defaults locked)
- HF Space UI simplified: single-panel layout, procedure selector cleaned up
- CI workflow updated: pinned action versions, pip caching, per-job concurrency control

### Fixed
- Inference pipeline dtype mismatch when switching between fp16 and bf16 checkpoints
- Per-procedure validation was silently skipped when `--validate` flag was absent

## [0.2.0] - 2025-06-01

### Added
- Brow lift procedure preset (PR #35, thanks @Deepak8858)
- Mentoplasty procedure preset (PR #36, thanks @P-r-e-m-i-u-m)
- Data-driven displacement model from real surgical data
- Clinical flags for edge case handling
- DisplacementModel class for fitted surgical displacements
- 6 new example scripts (evaluation, visualization, batch processing)
- Comprehensive API documentation
- GitHub wiki with 11 pages
- 200+ tracked issues for roadmap

### Changed
- Intensity parameter standardized to 0-100 scale
- Post-processing pipeline order: CodeFormer -> Real-ESRGAN -> histogram match -> sharpen -> blend
- Improved mask compositing with LAB skin tone matching

### Fixed
- SLURM config no longer hardcodes account names
- API docs now match actual code signatures
- Broken links in documentation index

## [0.1.0] - 2024-12-15

### Added
- Initial release
- 4 procedures: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic
- 4 inference modes: tps, img2img, controlnet, controlnet_ip
- MediaPipe 478-point face mesh landmark extraction
- Gaussian RBF landmark deformation
- ControlNet conditioning (CrucibleAI/ControlNetMediaPipeFace)
- Post-processing: CodeFormer, Real-ESRGAN, histogram matching
- ArcFace identity preservation check
- Gradio web demo
- CLI interface

[0.2.3]: https://github.com/dreamlessx/LandmarkDiff-public/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/dreamlessx/LandmarkDiff-public/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/dreamlessx/LandmarkDiff-public/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dreamlessx/LandmarkDiff-public/releases/tag/v0.1.0
