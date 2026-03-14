# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Brow lift procedure preset (contributed by [@Deepak8858](https://github.com/Deepak8858))
- Mentoplasty (chin surgery) procedure preset (contributed by [@P-r-e-m-i-u-m](https://github.com/P-r-e-m-i-u-m))
- Rathgeb et al. plastic surgery database for evaluation benchmarking
- CONTRIBUTORS.md with contribution log

### Fixed
- All example scripts updated to use correct API names and intensity scale (0-100)
- Batch inference script rewritten to use correct pipeline initialization
- CLI demo command now imports correct function from Gradio app
- API documentation aligned with actual code (DeformationHandle, PROCEDURE_LANDMARKS, render_wireframe)

### Planned
- FLUX.1-dev backbone upgrade
- IP-Adapter FaceID integration
- Additional procedure presets (otoplasty)
- Hugging Face Spaces demo
- Clinical validation study results

## [0.1.0] - 2026-03-13

### Added
- Core pipeline: MediaPipe 478-point landmark extraction, Gaussian RBF deformation, ControlNet conditioning, feathered mask compositing
- 4 surgical procedure presets: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic
- 3-mode inference pipeline: ControlNet, img2img, TPS
- Synthetic training pair generation via thin-plate spline warps
- 8 clinical photography augmentation types (lighting, color temp, JPEG, noise, distortion, blur, vignette, fluorescent)
- Clinical edge case handling: vitiligo, Bell's palsy, keloid-prone skin, Ehlers-Danlos syndrome
- Neural post-processing: CodeFormer face restoration, Real-ESRGAN super-resolution, ArcFace identity verification
- Evaluation harness: FID, LPIPS, SSIM, NME with Fitzpatrick skin type stratification (I-VI)
- Gradio web demo with 5 tabs: single prediction, multi-procedure comparison, intensity sweep, face analysis, multi-angle capture
- Face verifier module: 7-metric distortion detection with cascaded neural restoration
- 4-term training loss: diffusion MSE, landmark NME, ArcFace identity, LPIPS perceptual
- ControlNet fine-tuning script with BF16 mixed precision, EMA, cosine LR schedule
- SLURM job scripts for HPC training
- Apptainer container definition
- MICCAI 2026 paper source (Springer LNCS format)
- Comprehensive test suite (9 test modules)
- GPU training guide for SLURM clusters

[Unreleased]: https://github.com/dreamlessx/LandmarkDiff-public/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dreamlessx/LandmarkDiff-public/releases/tag/v0.1.0
