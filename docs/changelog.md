# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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

[0.2.0]: https://github.com/dreamlessx/LandmarkDiff-public/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dreamlessx/LandmarkDiff-public/releases/tag/v0.1.0
