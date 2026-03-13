# Contributing to LandmarkDiff

Thanks for your interest in contributing. This project is actively developed and we welcome contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LandmarkDiff-public.git
   cd LandmarkDiff-public
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests to make sure everything works:
   ```bash
   pytest tests/
   ```

## What We Need Help With

### High Priority
- **New procedure presets** - define anatomical displacement vectors for additional surgical procedures (e.g., mentoplasty, otoplasty, brow lift)
- **Clinical validation** - if you have access to pre/post-operative photo datasets, we'd love to collaborate
- **Multi-view consistency** - improving prediction quality across different face angles
- **3D extension** - incorporating FLAME or other 3D morphable models

### Medium Priority
- **Additional evaluation metrics** - domain-specific metrics for surgical outcome quality
- **Data augmentation** - new clinical photography degradation types
- **Documentation** - tutorials, examples, API docs

### Always Welcome
- Bug fixes
- Test coverage improvements
- Performance optimizations
- Typo fixes

## Development Workflow

1. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. Follow the existing code style:
   - Line length: 100 characters
   - Type hints where practical
   - Tests for new functionality

3. Run the linter and tests:
   ```bash
   ruff check landmarkdiff/ scripts/ tests/
   pytest tests/
   ```

4. Commit with a clear message:
   ```bash
   git commit -m "feat: add mentoplasty procedure preset"
   ```

5. Push and open a pull request.

## Code Structure

- `landmarkdiff/` - core library (landmarks, deformation, conditioning, inference)
- `landmarkdiff/synthetic/` - training data generation (TPS warps, augmentation)
- `scripts/` - CLI tools, training scripts, Gradio demo
- `tests/` - unit tests
- `paper/` - MICCAI manuscript source

## Adding a New Procedure

To add a new surgical procedure:

1. In `landmarkdiff/manipulation.py`, add control handles to `PROCEDURE_PRESETS`:
   ```python
   "mentoplasty": [
       ControlHandle(anchor_index=152, displacement=np.array([0, -8, 0]), radius=25.0),
       # ... more handles for chin advancement/setback
   ]
   ```

2. Add the procedure name to the CLI choices in `scripts/run_inference.py`

3. Add a test case in `tests/test_manipulation.py`

4. Document the landmark indices you used and the anatomical rationale

## Questions?

Open an issue. We're happy to discuss approaches before you start coding.
