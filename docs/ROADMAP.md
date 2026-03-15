# LandmarkDiff Roadmap

This document tracks planned milestones for LandmarkDiff. Timelines are estimates and may shift based on experimental results, reviewer feedback, and community input.

For discussion and feature requests, see [GitHub Discussions](https://github.com/dreamlessx/LandmarkDiff-public/discussions).

---

## v0.2.x (Current)

**Status:** Released

Core pipeline with TPS and ControlNet inference, six procedure presets, clinical edge-case handling, and Hugging Face Space demo.

- [x] MediaPipe 478-point face mesh extraction
- [x] Gaussian RBF deformation engine with procedure-specific presets
- [x] ControlNet-conditioned Stable Diffusion 1.5 generation
- [x] Neural post-processing (CodeFormer, Real-ESRGAN, ArcFace verification)
- [x] 6 procedures: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic, brow lift, mentoplasty
- [x] 4 inference modes: TPS (CPU), img2img, ControlNet, ControlNet+IP-Adapter
- [x] Clinical flags: vitiligo, Bell's palsy, keloid-prone skin, Ehlers-Danlos
- [x] Fitzpatrick-stratified evaluation protocol
- [x] Bilateral symmetry analysis with per-region scoring
- [x] Gradio demo with multi-angle capture and symmetry comparison
- [x] Docker and Apptainer container support
- [x] Hugging Face Space live demo (CPU)

---

## v0.3.0: Data-Driven Training

**Target:** Q3 2026

The focus of this release is moving from hand-tuned displacement vectors to displacements learned from real surgical outcome data, and completing the ControlNet fine-tuning pipeline.

### Data-driven displacement model
- [ ] Fit per-procedure displacement distributions from clinical before/after pairs
- [ ] Anatomically constrained sampling (region-specific variance, bilateral coherence)
- [ ] Validate against expert surgeon rankings on a held-out test set

### Training pipeline
- [ ] ControlNet fine-tuning on 50K+ synthetic TPS pairs (Phase A)
- [ ] Combined loss training on clinical pairs (Phase B): diffusion + landmark L2 + identity (ArcFace) + perceptual (LPIPS)
- [ ] Curriculum training: start with large deformations, anneal toward subtle corrections
- [ ] Multi-GPU DDP training with preemption checkpointing for HPC clusters

### Additional procedures
- [ ] Otoplasty (ear pinning) preset
- [ ] Genioplasty (sliding genioplasty) preset
- [ ] Community-contributed preset validation framework

### Evaluation
- [ ] Benchmark on the Rathgeb et al. plastic surgery database (CVPRW 2020)
- [ ] Populate quantitative results tables (FID, LPIPS, SSIM, NME, identity sim)
- [ ] Ablation studies: loss components, conditioning strategies, displacement models

### Publication
- [ ] MICCAI 2026 workshop paper submission (July 2026)
- [ ] arXiv preprint with supplementary materials

---

## v0.4.0: 3D Face Reconstruction

**Target:** Q4 2026

Move from single 2D images to full 3D face reconstruction from phone video. The user rotates their head while the phone captures from all angles (similar to Apple spatial audio or Face ID enrollment), and we reconstruct a textured 3D model for surgical deformation.

### Phone video capture pipeline
- [ ] Guided head-rotation capture flow (front-facing camera, ~15 second scan)
- [ ] Frame selection and quality filtering (blur, occlusion, lighting)
- [ ] Camera pose estimation from the video sequence

### 3D face reconstruction
- [ ] 3D face reconstruction from monocular video using FLAME 3D morphable model fitting
- [ ] Explore neural reconstruction approaches (NeRF, 3D Gaussian Splatting) for higher fidelity
- [ ] Textured mesh export (OBJ/glTF) with per-vertex color

### 3D surgical deformation
- [ ] 3D landmark extraction on the reconstructed mesh
- [ ] Surgical deformation applied directly in 3D space (vertex-level displacement)
- [ ] Multi-angle rendering of the predicted 3D result

### Interactive viewer
- [ ] WebGL/three.js 3D viewer in the Gradio demo
- [ ] Orbit controls for free-angle inspection of pre/post predictions

### Backbone upgrade
- [ ] FLUX.1-dev or SDXL backbone for higher quality generation at 1024x1024
- [ ] IP-Adapter FaceID v2 for stronger identity preservation
- [ ] LoRA fine-tuning support for domain-specific adaptation

---

## v0.5.0: Interactive 3D Surgical Preview

**Target:** Q1/Q2 2027

End-to-end mobile workflow: the patient captures a video scan on their phone and gets an interactive 3D preview they can rotate and inspect before committing to surgery.

### Physically-based 3D deformation
- [ ] Per-procedure surgical parameters controlling 3D mesh deformation
- [ ] Tissue property estimation from the reconstructed mesh (skin thickness, elasticity)
- [ ] Anatomically constrained deformation that respects bone structure and soft tissue mechanics

### Real-time interactive preview
- [ ] Patient rotates and inspects the 3D prediction on their phone in real time
- [ ] Before/after 3D comparison with measurement overlays (mm-scale changes)
- [ ] Video export: animated 360-degree rotation of the predicted result

### Mobile-optimized workflow
- [ ] Capture video -> reconstruct 3D -> apply deformation -> view prediction, all on phone
- [ ] Mesh compression and LOD for smooth rendering on mobile GPUs
- [ ] Offline-capable inference for clinic environments with limited connectivity

---

## v1.0.0: Clinical Validation

**Target:** 2027

Production-ready release with clinical validation data, 3D accuracy benchmarks, and regulatory groundwork.

### Clinical validation
- [ ] IRB-approved prospective study: compare predictions to actual surgical outcomes
- [ ] Inter-rater agreement study with board-certified plastic surgeons
- [ ] Statistical validation across Fitzpatrick types I-VI (equity audit)
- [ ] Calibration analysis: does predicted intensity correlate with actual surgical magnitude?
- [ ] Patient satisfaction studies: 3D preview vs actual surgical outcome

### 3D accuracy metrics
- [ ] Surface distance (Hausdorff, mean symmetric) between predicted and actual post-op scans
- [ ] Landmark reprojection error measured across multiple views
- [ ] Per-procedure accuracy breakdown with confidence intervals

### Regulatory pathway
- [ ] FDA 510(k) pathway exploration for clinical decision support classification
- [ ] FDA/regulatory considerations specific to 3D surgical visualization tools
- [ ] HIPAA-compliant deployment architecture (on-premise, no patient data leaves the facility)
- [ ] Documentation for SaMD (Software as a Medical Device) qualification

### Physics-informed modeling
- [ ] Finite element method (FEM) soft tissue simulation for physically plausible deformations
- [ ] Patient-specific tissue parameters estimated from skin elasticity measurements
- [ ] Scar formation prediction for incision planning

### Deployment
- [ ] Cloud deployment with Triton Inference Server
- [ ] React Native mobile app for standardized clinical photo capture
- [ ] DICOM integration for radiology workflows
- [ ] On-premise Docker stack for hospital IT environments

---

## How to Contribute

We welcome contributions at every level. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

- **New procedure presets**: define custom displacement vectors for procedures not yet supported
- **Clinical validation data**: if you have access to before/after surgical datasets, we would like to hear from you
- **Evaluation benchmarks**: run LandmarkDiff on your own data and share results
- **Feature requests**: open an issue or start a [Discussion](https://github.com/dreamlessx/LandmarkDiff-public/discussions)

Significant contributions earn co-authorship on the MICCAI 2026 paper. See the [Contributors table](../README.md#contributors) for recognition tiers.
