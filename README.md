# LandmarkDiff

Anatomically-conditioned latent diffusion for photorealistic facial surgery outcome prediction from standard clinical photography.

LandmarkDiff takes a patient's pre-operative photograph and a specified surgical procedure, then generates a photorealistic prediction of post-operative facial appearance. It works by deforming MediaPipe Face Mesh landmarks according to procedure-specific anatomical displacement vectors, then conditioning a Stable Diffusion 1.5 backbone via ControlNet to synthesize the predicted face while preserving patient identity.

> **Paper:** "LandmarkDiff: Anatomically-Conditioned Latent Diffusion for Photorealistic Facial Surgery Outcome Prediction," targeting MICCAI 2026.

## Supported Procedures

| Procedure | Region | Key landmarks |
|-----------|--------|--------------|
| Rhinoplasty | Nasal bridge, tip, alar base | 48-51, 4-6, 195-196 |
| Blepharoplasty | Upper/lower eyelids | 159-160, 386-387, 33, 133, 362, 263 |
| Rhytidectomy | Jawline, cheeks, periauricular | 234, 454, 132, 361, 172, 397 |
| Orthognathic | Mandible, maxilla, chin | 152, 175, 18, 200 |

## Architecture

```
Input Photo → MediaPipe 478-point extraction
                    ↓
              Gaussian RBF deformation (procedure-specific)
                    ↓
              Tessellation mesh rendering (2556-edge wireframe)
                    ↓
              ControlNet (CrucibleAI/ControlNetMediaPipeFace)
                    ↓
              Stable Diffusion 1.5 (latent space generation)
                    ↓
              Feathered mask compositing (LAB-space skin tone matching)
                    ↓
              Post-processing (CodeFormer + Real-ESRGAN + ArcFace verify)
                    ↓
              Predicted post-operative face
```

## Quick Start

### Installation

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public
pip install -e .

# For training
pip install -e ".[train]"

# For Gradio demo
pip install -e ".[app]"
```

### Run Inference

```bash
python scripts/run_inference.py /path/to/face.jpg \
    --procedure rhinoplasty \
    --intensity 0.6 \
    --mode controlnet
```

### Launch Gradio Demo

```bash
python scripts/app.py
```

The demo includes:
- Single prediction with adjustable intensity
- Intensity sweep across 0-100%
- Side-by-side before/after comparison
- Clinical flags (vitiligo, Bell's palsy, keloid, Ehlers-Danlos)
- Multi-angle guided capture (frontal, 3/4, profile views)

### Generate Synthetic Training Data

```bash
# Download face images (FFHQ)
python scripts/download_ffhq.py --num 5000 --resolution 512

# Generate landmark-deformed training pairs
python scripts/generate_synthetic_data.py \
    --input data/ffhq_samples/ \
    --output data/synthetic_pairs/ \
    --num 5000
```

### Train ControlNet

```bash
# Single GPU
python scripts/train_controlnet.py \
    --data_dir data/synthetic_pairs/ \
    --output_dir checkpoints/ \
    --num_train_steps 10000

# SLURM (A100 80GB recommended)
sbatch scripts/train_slurm.sh
```

See [docs/GPU_TRAINING_GUIDE.md](docs/GPU_TRAINING_GUIDE.md) for detailed HPC setup instructions.

### Docker

```bash
docker build -t landmarkdiff .
docker compose up landmarkdiff
# Open http://localhost:7860
```

### Make targets

```bash
make help          # Show all available commands
make install       # Install (inference only)
make install-all   # Install everything
make test          # Run tests
make lint          # Run linter
make demo          # Launch Gradio demo
make train         # Train ControlNet
make paper         # Build MICCAI paper PDF
make docker        # Build Docker image
```

## Project Structure

```
landmarkdiff/               # Core library
    landmarks.py            #   MediaPipe 478-point face mesh extraction
    conditioning.py         #   ControlNet conditioning (tessellation + Canny)
    manipulation.py         #   Gaussian RBF landmark deformation
    masking.py              #   Feathered surgical mask generation
    inference.py            #   Full pipeline (ControlNet / img2img / TPS modes)
    losses.py               #   Combined loss (diffusion + landmark + identity + perceptual)
    evaluation.py           #   Metrics (FID, LPIPS, SSIM, NME, Fitzpatrick ITA)
    clinical.py             #   Clinical edge cases (vitiligo, Bell's palsy, keloid)
    postprocess.py          #   Face restoration (CodeFormer, GFPGAN, Real-ESRGAN)
    synthetic/
        pair_generator.py   #   Training pair generation pipeline
        tps_warp.py         #   Thin-plate spline warping
        augmentation.py     #   Clinical photography augmentations

scripts/                    # CLI tools and job scripts
    app.py                  #   Gradio web demo (5 tabs)
    train_controlnet.py     #   ControlNet fine-tuning
    evaluate.py             #   Automated evaluation harness
    demo.py                 #   CLI demo with visualizations
    ...                     #   Data download, pair generation, SLURM scripts

examples/                   # Example scripts for common tasks
benchmarks/                 # Performance benchmarks
paper/                      # MICCAI 2026 manuscript (Springer LNCS)
docs/                       # Documentation, tutorials, API reference
    tutorials/              #   Quick start, custom procedures, training, deployment
    api/                    #   Module-level API docs
configs/                    # Training configuration (YAML)
containers/                 # Apptainer/Singularity definition
tests/                      # Unit tests (9 test modules)
demos/                      # Sample output images
```

See [docs/](docs/) for full documentation, [examples/](examples/) for runnable scripts, and [benchmarks/](benchmarks/) for performance data.

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| FID | Frechet Inception Distance (realism) | < 50 |
| LPIPS | Learned Perceptual Image Patch Similarity | < 0.15 |
| SSIM | Structural Similarity Index | > 0.80 |
| NME | Normalized Mean Error (landmark accuracy) | < 0.05 |
| Identity Sim | ArcFace cosine similarity (identity preservation) | > 0.85 |

All metrics are stratified by Fitzpatrick skin type (I-VI) to ensure equitable performance across skin tones.

## Clinical Edge Cases

LandmarkDiff includes specialized handling for:
- **Vitiligo**: LAB-based patch detection with mask intensity reduction
- **Bell's palsy**: Disables deformation on the paralyzed side
- **Keloid-prone skin**: Softens mask transitions in keloid-prone regions
- **Ehlers-Danlos syndrome**: Wider influence radii for hypermobile tissue

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA
- ~6GB VRAM for inference (SD1.5 + ControlNet)
- A100 80GB recommended for training
- MediaPipe for face landmark detection

## Roadmap

### Current (v0.1 - Spring 2026)
- [x] Core pipeline: landmark extraction, RBF deformation, ControlNet conditioning, mask compositing
- [x] 4 procedure presets (rhinoplasty, blepharoplasty, rhytidectomy, orthognathic)
- [x] Synthetic training pair generation via TPS warps
- [x] Clinical edge case handling (vitiligo, Bell's palsy, keloid, Ehlers-Danlos)
- [x] Neural post-processing (CodeFormer, Real-ESRGAN, ArcFace identity verification)
- [x] Gradio demo with multi-angle capture
- [ ] ControlNet fine-tuning on 50K+ synthetic pairs (in progress)
- [ ] Populate results tables in paper

### Next (v0.2 - Summer 2026)
- [ ] FLUX.1-dev backbone upgrade (higher quality generation at 1024x1024)
- [ ] IP-Adapter FaceID for stronger identity preservation
- [ ] Additional procedure presets (mentoplasty, brow lift, otoplasty)
- [ ] Clinical validation with board-certified plastic surgeons
- [ ] Hugging Face interactive demo deployment
- [ ] arXiv preprint (target: April 2026)

### Future (v1.0)
- [ ] FLAME 3D morphable model integration for depth-aware deformation
- [ ] Multi-view consistency loss across frontal/profile predictions
- [ ] Physics-informed tissue simulation (FEM for soft tissue response)
- [ ] React Native mobile capture app with standardized clinical photo acquisition
- [ ] Cloud deployment with Triton inference server

### Publication targets
- MICCAI 2026 workshop paper (July 2026 submission)
- RSNA 2026 abstract (May 2026)
- Full conference paper (CVPR/NeurIPS 2027)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@inproceedings{landmarkdiff2026,
  title={LandmarkDiff: Anatomically-Conditioned Latent Diffusion for Photorealistic Facial Surgery Outcome Prediction},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year={2026},
  publisher={Springer}
}
```

## Contributors

We track all contributions and contributors will be acknowledged in the MICCAI 2026 paper. Significant contributions earn co-authorship.

| Contribution Level | Recognition |
|---|---|
| Bug fix / typo | Acknowledged in README |
| New procedure preset | Acknowledged in paper + README |
| Feature module (e.g., new loss, metric) | Co-author on paper |
| Clinical validation data | Co-author on paper |
| Sustained multi-feature contributions | Co-author on paper |

### Current Contributors

| GitHub Handle | Contribution |
|---|---|
| [@dreamlessx](https://github.com/dreamlessx) | Core architecture, training pipeline, paper |

To join this list, open a PR or contribute to an issue. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contributing

Contributions welcome. Please open an issue or pull request.

For major changes, please open an issue first to discuss the proposed approach.

## Acknowledgments

- [CrucibleAI](https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace) for the MediaPipe Face ControlNet
- [MediaPipe](https://google.github.io/mediapipe/) for face mesh extraction
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet)
