# FAQ and Troubleshooting

## General

**What input does LandmarkDiff need?**

A single 2D photograph of a face. No 3D scans, no depth sensors. A clear frontal photo with even lighting works best. The image gets resized to 512x512 internally.

**Can I run this without a GPU?**

Yes, in TPS mode. It does geometric warping on CPU in about 0.5 seconds. The photorealistic diffusion modes (controlnet, img2img) need a GPU with at least 6GB VRAM.

**What GPU do I need?**

For inference: anything with 6GB+ VRAM (RTX 3060 and up, T4, etc.). For training: 24GB minimum (RTX 3090), 80GB recommended (A100). See the full GPU requirements table below.

| Mode | Min VRAM | Recommended | Notes |
|------|----------|-------------|-------|
| tps | 0 GB | -- | CPU-only, no GPU needed |
| img2img | 4 GB | 6 GB | SD1.5 with model_cpu_offload |
| controlnet | 6 GB | 8 GB | SD1.5 + ControlNet |
| controlnet_ip | 8 GB | 10 GB | SD1.5 + ControlNet + IP-Adapter |
| Training | 24 GB | 80 GB | RTX 3090 minimum, A100 80GB recommended |

With `enable_model_cpu_offload()`, CUDA VRAM usage is reduced by moving model components to CPU RAM when not in use. This is enabled by default.

On Apple Silicon (MPS), all models run in fp32 due to MPS backend limitations. This roughly doubles memory compared to CUDA fp16.

**Is this FDA approved for clinical use?**

No. This is a research tool. Predictions are simulations, not medical advice. Any clinical deployment would need regulatory review.

**Can I use my own photos?**

Yes. The pipeline works on any photo with a detectable face. Upload through the Gradio demo or pass a file path to the CLI.

**What image formats are supported?**

Any format readable by OpenCV: JPEG, PNG, BMP, TIFF, WebP. The input is resized to 512x512 internally. For best results, use well-lit clinical photos with the face centered and filling most of the frame. PNG with transparency (alpha channel) is handled by discarding the alpha channel before processing.

**How does the intensity parameter work?**

Intensity is always on a 0-100 scale:

| Value | Effect |
|-------|--------|
| 0 | No deformation (identity transform) |
| 33 | Mild surgical effect |
| 50 | Moderate (default in most examples) |
| 66 | Noticeable |
| 100 | Maximum / aggressive |

Internally, `apply_procedure_preset()` divides intensity by 100 to get a scale factor that multiplies all displacement vectors. For example, a rhinoplasty tip displacement of `(0, -2.0)` at intensity=50 becomes `(0, -2.0 * 0.5) = (0, -1.0)` pixels.

When using the `DisplacementModel` (data-driven mode), the pipeline maps 0-100 to a 0-2.0 range where 1.0 represents the average observed displacement from real surgery pairs. So intensity=50 gives average displacement, and intensity=100 gives 2x average.

**What is `pixel_coords` and why can't I call it?**

`pixel_coords` is a `@property` on the `FaceLandmarks` dataclass, not a method. Access it without parentheses:

```python
face = extract_landmarks(image)

# Correct
coords = face.pixel_coords     # returns (478, 2) numpy array

# Wrong: will raise TypeError
coords = face.pixel_coords()   # pixel_coords is not callable
```

The property computes pixel coordinates from the normalized (0-1) MediaPipe coordinates using the image dimensions stored in the `FaceLandmarks` object.

**Can I process multiple images in a batch?**

Yes. Use the batch inference script:

```bash
python examples/batch_inference.py /path/to/image_dir/ \
    --procedure blepharoplasty \
    --intensity 50 \
    --output output/batch/
```

Or in Python, create the pipeline once and loop over images:

```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()

for image_path in image_paths:
    result = pipeline.generate(load_image(image_path), procedure="rhinoplasty", intensity=60)
```

Creating the pipeline once and reusing it avoids reloading model weights on every call.

**Is the pipeline thread-safe?**

No. The underlying PyTorch and diffusion models are not thread-safe. If you need concurrent processing, use separate processes (e.g., `multiprocessing`) with each process holding its own pipeline instance. For a web server, use a single pipeline instance per worker process and route requests sequentially within each worker.

## Installation Issues

**`mediapipe` fails to install**

MediaPipe has specific Python version requirements. Make sure you're on Python 3.10 or 3.11. Python 3.12 support depends on the mediapipe version.

```bash
# check your python version
python --version

# if needed, create a fresh env
conda create -n landmarkdiff python=3.11
conda activate landmarkdiff
pip install -e .
```

**CUDA out of memory**

The full pipeline uses about 5.2GB VRAM. If you're running out of memory:

1. Close other GPU processes (`nvidia-smi` to check)
2. Use `--mode tps` for CPU-only inference
3. Reduce `--steps` (fewer diffusion steps = less memory)
4. Make sure you're using fp16 (the default)

**`torch` not finding CUDA**

```bash
# check if pytorch sees your GPU
python -c "import torch; print(torch.cuda.is_available())"

# if False, reinstall pytorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Models downloading every time**

The models cache to `~/.cache/huggingface/`. If that's getting cleared (e.g. on a cluster with temp home dirs), set `HF_HOME` to a persistent location:

```bash
export HF_HOME=/path/to/persistent/cache
```

## Pipeline Issues

**"No face detected"**

MediaPipe couldn't find a face. Common causes:
- Image is too dark or too bright
- Face is heavily occluded (sunglasses, mask)
- Face is at an extreme angle (>60 degrees from frontal)
- Image resolution is very low

Try a clearer, more frontal photo.

**Output looks like a different person**

This can happen when the ControlNet generates from the mesh without enough identity signal from the input. Try:
- Using `controlnet_ip` mode (adds IP-Adapter for identity preservation)
- Lowering `controlnet_conditioning_scale` (less mesh influence, more identity)
- Checking the ArcFace identity score in the output dict

**Output has visible seams or color mismatch**

The Laplacian pyramid blending should handle this, but if you see artifacts:
- Make sure post-processing is enabled (`postprocess=True`)
- The LAB histogram matching step corrects skin tone differences
- Check if the surgical mask is too large or too small for the procedure

**TPS warp looks distorted**

TPS mode is purely geometric: it pushes pixels around without any neural generation. At high intensity (>70%) the distortion becomes obvious. This is expected. Use ControlNet mode for photorealistic results.

## Training Issues

**Training loss is NaN**

Common causes:
- Using fp16 instead of bf16 (fp16 can overflow with diffusion models)
- Learning rate too high (start with 1e-5)
- Bad training data (corrupted images or landmarks)

Make sure `mixed_precision: bf16` is set in your config.

**Training is very slow**

- Check that you're actually using the GPU (`nvidia-smi` during training)
- Pre-compute TPS warps before training to avoid CPU bottleneck
- Use gradient accumulation to increase effective batch size without more VRAM
- See the [GPU training guide](GPU_TRAINING_GUIDE.md) for SLURM and multi-GPU setup

**Checkpoints are huge**

ControlNet checkpoints are about 1.4GB each in fp16. Use safetensors format (the default) which is slightly smaller and faster to load. Set `save_every_n_steps` in the config to avoid saving too frequently.

## Gradio Demo Issues

**Demo won't start**

```bash
# make sure gradio is installed
pip install -e ".[app]"

# check the port isn't in use
lsof -i :7860

# try a different port
python scripts/app.py --port 7861
```

**Demo is slow on first run**

The first inference downloads model weights (~6GB total). Subsequent runs use the cached models and should be much faster.

**File upload not working in Colab**

The Gradio demo uses `share=True` by default in Colab, which creates a public URL. If file upload fails, try running locally or using the notebook instead.

## Developer Issues

**mypy reports type errors in LandmarkDiff modules**

The project uses mypy with `--ignore-missing-imports` because several dependencies (mediapipe, insightface, lpips, etc.) do not ship type stubs. Common mypy errors and fixes:

1. **`Module has no attribute`** for optional imports: Some modules are imported conditionally with try/except. mypy cannot track these. The `pyproject.toml` already includes `[[tool.mypy.overrides]]` entries for known modules.

2. **`Incompatible types in assignment`** with numpy: The `pixel_coords` property returns `np.ndarray` but mypy sometimes infers a different type. Use explicit type annotations:
   ```python
   coords: np.ndarray = face.pixel_coords
   ```

3. **`Cannot find implementation or library stub`**: Install type stubs:
   ```bash
   pip install types-Pillow types-PyYAML
   ```

Run the type checker the same way CI does:
```bash
mypy landmarkdiff/ --ignore-missing-imports
```

**pre-commit hooks fail**

The project uses pre-commit with ruff (lint + format) and mypy. Common failures:

1. **ruff format**: Code was not formatted. Fix:
   ```bash
   ruff format landmarkdiff/ scripts/ tests/
   ```

2. **ruff check**: Linting errors. Fix auto-fixable ones:
   ```bash
   ruff check --fix landmarkdiff/ scripts/ tests/
   ```

3. **trailing-whitespace or end-of-file-fixer**: These hooks auto-fix the issue. Just re-stage the files and commit again.

4. **check-added-large-files**: You are trying to commit a file larger than 1MB. Use Git LFS for model checkpoints and large data files.

5. **mypy**: Type errors. Run `mypy landmarkdiff/ --ignore-missing-imports` to see the full report and fix errors before committing.

To install pre-commit hooks after cloning:
```bash
pip install -e ".[dev]"
pre-commit install
```

To run all hooks manually on all files:
```bash
pre-commit run --all-files
```

**Docker build fails**

Common Docker build issues:

1. **CUDA version mismatch**: The GPU Dockerfile (`Dockerfile`) uses `nvidia/cuda:12.1.1-devel-ubuntu22.04`. If your host has a different CUDA driver version, ensure driver compatibility (CUDA 12.1 requires driver >= 530).

2. **Out of disk space**: The GPU image is large (~10 GB). Clean old images:
   ```bash
   docker system prune -a
   ```

3. **pip install fails inside container**: Network issues or missing system packages. The Dockerfiles install the required system libraries (`libgl1-mesa-glx`, `libglib2.0-0`, etc.). If you modified the Dockerfile and see `ImportError: libGL.so.1`, add:
   ```dockerfile
   RUN apt-get update && apt-get install -y libgl1-mesa-glx
   ```

4. **CPU Dockerfile (`Dockerfile.cpu`)** installs CPU-only PyTorch from `https://download.pytorch.org/whl/cpu`. If you need GPU support, use the main `Dockerfile` instead.

5. **Permission errors with mounted volumes**: The container runs as root by default. If you mount host directories, ensure the directories exist and are writable:
   ```bash
   mkdir -p data checkpoints
   docker compose up app
   ```

**Sphinx documentation build fails**

1. **Missing dependencies**: Install the docs requirements:
   ```bash
   pip install -r docs/requirements.txt
   ```
   Required packages: `sphinx>=7.0`, `furo`, `myst-parser`, `sphinx-autodoc-typehints`, `sphinx-copybutton`.

2. **MyST markdown parse errors**: The docs use MyST-flavored markdown. Common issues:
   - Directive syntax uses `{toctree}` not `:toctree:` in code fences.
   - Heading levels must not skip (e.g., going from `#` to `###` without `##` in between).
   - Indentation inside directives must be consistent.

3. **autodoc import errors**: Sphinx tries to import `landmarkdiff` to generate API docs. Install the package first:
   ```bash
   pip install -e .
   sphinx-build -b html docs/ docs/_build/html
   ```

4. **Build locally with Make or Docker:**
   ```bash
   # Make
   make docs

   # Docker compose
   docker compose run docs
   ```

   The HTML output goes to `docs/_build/html/`.
