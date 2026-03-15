# Contributing

Thanks for your interest in contributing to LandmarkDiff. This project is actively developed and we welcome contributions of all kinds.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LandmarkDiff-public.git
   cd LandmarkDiff-public
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests to verify everything works:
   ```bash
   pytest tests/
   ```

## What We Need Help With

### High Priority
- **New procedure presets**: define displacement vectors for additional surgical procedures (e.g., otoplasty, lip augmentation)
- **Clinical validation**: if you have access to pre/post-operative photo datasets, we would love to collaborate
- **Multi-view consistency**: improving prediction quality across different face angles

### Medium Priority
- **Evaluation metrics**: domain-specific metrics for surgical outcome quality
- **Data augmentation**: new clinical photography degradation types
- **Performance**: faster inference, reduced memory usage, batched processing

### Always Welcome
- Bug fixes
- Test coverage improvements
- Documentation improvements
- Typo corrections

## Development Workflow

1. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. Follow the existing code style:
   - Line length: 100 characters (enforced by ruff)
   - Type hints on function signatures
   - Docstrings for public functions
   - Tests for new functionality

3. Run the linter, type checker, and tests:
   ```bash
   ruff check landmarkdiff/ scripts/ tests/
   mypy landmarkdiff/ --ignore-missing-imports
   pytest tests/
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "feat: add otoplasty procedure preset"
   ```

   Commit message prefixes we use:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `test:` adding or fixing tests
   - `refactor:` code restructuring
   - `perf:` performance improvement

5. Push and open a pull request.

## Adding a New Procedure

This is the most common contribution. Here is the full process:

### Step 1: Identify Landmarks

Look up the MediaPipe Face Mesh landmark indices for the target anatomy. The [MediaPipe face mesh documentation](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) has an interactive 3D viewer. You can also use `python -m landmarkdiff landmarks face.jpg` to visualize the mesh on a real face.

### Step 2: Define Landmarks and Displacement

In `landmarkdiff/manipulation.py`:

```python
PROCEDURE_LANDMARKS["otoplasty"] = [127, 234, 356, 454, ...]
PROCEDURE_RADIUS["otoplasty"] = 25.0  # choose based on region size
```

Then in `_get_procedure_handles()`, add an `elif procedure == "otoplasty":` block with anatomically-motivated displacement vectors.

Key rules:
- Displacements are in pixels at 512x512 resolution
- Use `scale` (0-1) to multiply all displacements
- Left/right symmetry: left landmarks move left (-X), right move right (+X) for narrowing, and vice versa for widening
- Upward in image space is -Y
- Use smaller influence radii (0.5x-0.7x) for fine structures, larger (1.0x-1.2x) for broad effects

### Step 3: Add Mask Config

In `landmarkdiff/masking.py`, add your procedure to `MASK_CONFIG`:

```python
MASK_CONFIG["otoplasty"] = {
    "landmark_indices": [127, 234, 356, 454, ...],
    "dilation_px": 20,       # morphological dilation
    "feather_sigma": 12.0,   # Gaussian feathering sigma
}
```

### Step 4: Add CLI Support

In `landmarkdiff/__main__.py`, add the procedure to the choices list.

### Step 5: Add a Prompt (Optional)

In `landmarkdiff/inference.py`, add to `PROCEDURE_PROMPTS`:

```python
PROCEDURE_PROMPTS["otoplasty"] = (
    "clinical photograph, patient face, natural ears, ..."
)
```

### Step 6: Write Tests

Add test cases in `tests/test_manipulation.py` to verify your deformation produces reasonable results.

## Code Structure

| Directory | Contents |
|-----------|----------|
| `landmarkdiff/` | Core library: landmarks, deformation, conditioning, inference |
| `landmarkdiff/synthetic/` | Training data generation (TPS warps, augmentation) |
| `scripts/` | CLI tools, training scripts, Gradio demo, evaluation |
| `tests/` | Unit and integration tests |
| `configs/` | YAML training configs |
| `paper/` | Manuscript source |
| `docs/` | Additional documentation |
| `containers/` | Docker and container configs |

## Testing

Tests run with pytest:

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=landmarkdiff --cov-report=html

# Specific module
pytest tests/test_manipulation.py -v
```

Most tests mock heavy dependencies (MediaPipe, PyTorch, diffusers) so they run on CPU without model downloads.

## Pull Request Process

1. Ensure all CI checks pass (lint, type-check, tests on Python 3.10/3.11/3.12)
2. Update docstrings for any changed public API
3. Add yourself to CONTRIBUTORS.md if it is your first contribution
4. PRs are typically reviewed within a few days

## Questions?

Open an [issue](https://github.com/dreamlessx/LandmarkDiff-public/issues) or start a [discussion](https://github.com/dreamlessx/LandmarkDiff-public/discussions). Happy to talk through approaches before you start coding.
