# Contributing to LandmarkDiff

Thanks for your interest in contributing. Whether you're fixing a typo, adding a new procedure preset, or building a whole new module, this guide will help you get set up and moving quickly.

If you have questions that this guide doesn't answer, feel free to open a [Discussion](https://github.com/dreamlessx/LandmarkDiff-public/discussions).

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Adding a New Procedure Preset](#adding-a-new-procedure-preset)
- [Adding Clinical Flags](#adding-clinical-flags)
- [Documentation](#documentation)
- [Issue Labels](#issue-labels)
- [Recognition](#recognition)
- [Community Guidelines](#community-guidelines)

---

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git
- A virtual environment tool (venv, conda, etc.)
- GPU with 6GB+ VRAM is helpful but not required (TPS mode runs on CPU)

### Clone and install

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install with all dev dependencies
pip install -e ".[dev]"
```

The `[dev]` extra pulls in pytest, ruff, mypy, and everything else you need for local development. If you also want training, eval, or Gradio dependencies:

```bash
# Everything at once
pip install -e ".[train,eval,app,dev]"
```

### Pre-commit hooks

We use pre-commit to catch lint and format issues before they reach CI:

```bash
pip install pre-commit
pre-commit install
```

After this, `ruff check` and `ruff format --check` will run automatically on every commit.

### Verify your setup

```bash
make check
```

This runs lint, type checking, and the test suite. If everything passes, you're ready.

---

## Code Style

### Linting and formatting

We use [ruff](https://github.com/astral-sh/ruff) for both linting and formatting.

- **Line length:** 100 characters
- **Target Python:** 3.10
- **Enabled rules:** E (pycodestyle), F (pyflakes), I (isort), N (pep8-naming), UP (pyupgrade), B (bugbear), SIM (simplify)

```bash
# Check for lint issues
ruff check landmarkdiff/ scripts/ tests/

# Auto-format
ruff format landmarkdiff/ scripts/ tests/

# Auto-fix lint issues where possible
ruff check --fix landmarkdiff/ scripts/ tests/
```

Or use the Makefile shortcuts:

```bash
make lint      # check only
make format    # auto-fix + format
```

### Type checking

We use [mypy](https://mypy-lang.org/) for static type analysis:

```bash
mypy landmarkdiff/ --ignore-missing-imports
# or
make type-check
```

Type annotations are not required everywhere, but we appreciate them on public function signatures. If you add a new module, make sure mypy doesn't introduce new errors.

### General conventions

- Use `from __future__ import annotations` at the top of new modules for modern annotation syntax.
- Use `TYPE_CHECKING` guards for imports that are only needed for type hints.
- Docstrings follow Google style (compatible with sphinx napoleon).
- Keep files under 500 lines when practical.

---

## Testing

Tests live in the `tests/` directory and use [pytest](https://docs.pytest.org/).

### Running tests

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=landmarkdiff --cov-report=term-missing

# Skip slow tests (GPU, large data)
pytest tests/ -v -m "not slow"
```

Or:

```bash
make test       # full suite
make test-fast  # skip slow tests
```

### Writing tests

- Test files go in `tests/` and should be named `test_<module>.py`.
- Use plain pytest assertions. No need for `unittest.TestCase`.
- Mock external dependencies (MediaPipe, PyTorch models, file I/O) so tests run without a GPU.
- If your test needs a GPU or takes more than a few seconds, mark it with `@pytest.mark.slow`.
- Each new feature or bug fix should come with at least one test that exercises the change.

Example test structure:

```python
"""Tests for the frobnication module."""

import numpy as np
import pytest

from landmarkdiff.frobnication import frobnicate


class TestFrobnicate:
    def test_basic(self):
        result = frobnicate(np.zeros(10))
        assert result.shape == (10,)

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            frobnicate(np.array([]))

    @pytest.mark.slow
    def test_gpu_path(self):
        # Only runs when slow tests are enabled
        ...
```

### CI

CI runs on every push to `main` and on every pull request. It checks:

1. **Lint** -- `ruff check` and `ruff format --check`
2. **Type check** -- `mypy`
3. **Tests** -- `pytest` on Python 3.10, 3.11, and 3.12

All three must pass before a PR can be merged.

---

## Submitting Changes

### Workflow

1. **Fork** the repository on GitHub.
2. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b add-otoplasty-preset
   ```
3. **Make your changes.** Write code, add tests, update docs as needed.
4. **Run the checks locally:**
   ```bash
   make check  # or: ruff check ... && mypy ... && pytest ...
   ```
5. **Commit** with a clear message. We don't enforce a rigid format, but try to be descriptive:
   ```
   Add otoplasty procedure preset

   Targets 15 ear landmarks with lateral displacement for
   ear pinning simulation. Includes unit tests and docs.
   ```
6. **Push** your branch and **open a pull request** against `main`.
7. Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md). The checklist will remind you of the standard checks.

### What to expect

- A maintainer will review your PR, usually within a few days.
- CI will run automatically. If it fails, check the logs and push a fix.
- We may suggest changes. This is normal and not a rejection. Iterate until everyone is happy.
- Once approved, a maintainer will merge your PR.

### Tips

- For major changes (new modules, architectural decisions, new loss functions), open an issue first to discuss the approach. This avoids wasted effort if the direction doesn't align with the project goals.
- Small, focused PRs are easier to review than large ones. If your change touches multiple areas, consider splitting it up.
- If your PR adds a new procedure preset, the review will check anatomical plausibility of the displacement vectors in addition to code quality.

---

## Adding a New Procedure Preset

New procedure presets are one of the most impactful contributions. Two community contributors have already added brow lift and mentoplasty this way. Here is the step-by-step process.

### 1. Research the procedure

Understand which anatomical structures are affected and in what directions. Look at surgical textbooks, published anthropometric data, or before/after imagery to understand the typical tissue displacement patterns.

### 2. Identify landmarks

Find the relevant landmark indices in the [MediaPipe 478-point face mesh](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png). You can also use the landmark visualization example to see all 478 points on a test face:

```bash
python examples/landmark_visualization.py /path/to/face.jpg
```

### 3. Add the preset to `landmarkdiff/manipulation.py`

You need to touch three data structures:

**a) `PROCEDURE_LANDMARKS`** -- add your landmark index list:

```python
PROCEDURE_LANDMARKS: dict[str, list[int]] = {
    # ... existing procedures ...
    "otoplasty": [
        # ear landmarks
        234, 93, 132, ...
    ],
}
```

**b) `PROCEDURE_RADIUS`** -- set the Gaussian RBF influence radius (pixels at 512x512). Smaller radius (15-20px) for fine structures, larger (35-40px) for broad tissue mobilization:

```python
PROCEDURE_RADIUS: dict[str, float] = {
    # ... existing ...
    "otoplasty": 20.0,
}
```

**c) `_get_procedure_handles()`** -- define the displacement vectors for each landmark. Each displacement is a `(dx, dy)` pair in pixels (at 512x512 resolution), scaled by the intensity parameter. Positive x is rightward, positive y is downward:

```python
def _get_procedure_handles(procedure, indices, scale, radius):
    # ... existing cases ...
    elif procedure == "otoplasty":
        for idx in indices:
            dx, dy = _otoplasty_displacement(idx)
            handles.append(DeformationHandle(
                landmark_index=idx,
                displacement=np.array([dx * scale, dy * scale]),
                influence_radius=radius,
            ))
```

### 4. Register in the CLI

Add your procedure name to the choices list in `landmarkdiff/__main__.py` so the CLI recognizes it.

### 5. Add tests

Create or extend a test file in `tests/`. At minimum, test:

- The preset applies without errors at various intensity levels (0, 50, 100).
- The output landmarks differ from the input (something actually moved).
- Only the expected landmarks change significantly.
- The landmark count is preserved (478 in, 478 out).

```python
def test_otoplasty_preset():
    face = make_dummy_face()
    result = apply_procedure_preset(face, "otoplasty", intensity=60)
    assert result.landmarks.shape == face.landmarks.shape
    assert not np.allclose(result.landmarks, face.landmarks)
```

### 6. Update documentation

- Add a section to `README.md` under "Supported Procedures" following the existing format (description, landmark indices, influence radius).
- If you used published references for the displacement vectors, cite them.

### 7. Open a PR

Use the "New Procedure Preset" type in the PR template checklist. Mention the anatomical rationale in your PR description.

See PRs [#35](https://github.com/dreamlessx/LandmarkDiff-public/pull/35) (brow lift) and [#36](https://github.com/dreamlessx/LandmarkDiff-public/pull/36) (mentoplasty) for real examples of successful procedure contributions.

---

## Adding Clinical Flags

Clinical flags modify how deformations and masks behave for patients with specific conditions. The existing flags (vitiligo, Bell's palsy, keloid-prone skin, Ehlers-Danlos) are defined in `landmarkdiff/clinical.py`.

To add a new clinical flag:

1. **Add the field** to the `ClinicalFlags` dataclass in `landmarkdiff/clinical.py`.
2. **Implement the modifier** as a function in the same module. The modifier should adjust either the deformation handles, the mask, or the influence radii depending on the condition.
3. **Wire it in.** Call your modifier from `apply_procedure_preset()` in `manipulation.py` (for deformation changes) or from the mask generation code in `masking.py` (for mask changes).
4. **Add the flag to the Gradio demo** in `scripts/app.py` -- typically as a checkbox in Tab 1.
5. **Write tests** that verify the flag actually modifies the output relative to the unflagged case.
6. **Document** the clinical rationale. Include references to the medical literature if possible.

---

## Documentation

Documentation is built with [Sphinx](https://www.sphinx-doc.org/) using the [Furo](https://pradyunsg.me/furo/) theme and [MyST](https://myst-parser.readthedocs.io/) for Markdown support.

### Building docs locally

```bash
# Install doc dependencies
pip install -r docs/requirements.txt

# Build HTML docs
cd docs
sphinx-build -b html . _build/html

# Open in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Doc conventions

- Docs are written in Markdown (MyST), not reStructuredText.
- API reference docs are auto-generated from docstrings via `sphinx.ext.autodoc`.
- Tutorials go in `docs/tutorials/`.
- API docs go in `docs/api/`.
- Use Google-style docstrings in Python code so napoleon can parse them.

### When to update docs

- New public API (class, function, module): add or update the relevant API doc.
- New procedure or clinical flag: update the README and add a tutorial if the feature is complex.
- Changed behavior: update any docs that describe the old behavior.

---

## Issue Labels

| Label | Meaning |
|-------|---------|
| `bug` | Something broken or producing wrong results |
| `enhancement` | New feature or improvement to existing functionality |
| `new-procedure` | Proposal or implementation of a new surgical procedure preset |
| `documentation` | Docs-only changes |
| `good first issue` | Suitable for newcomers to the project |
| `help wanted` | Maintainer is looking for community input or implementation |
| `ci` | CI/CD pipeline changes |
| `question` | Needs discussion, not necessarily a code change |

---

## Recognition

We track all contributions and contributors are acknowledged in the project:

| Contribution Level | Recognition |
|---|---|
| Bug fix or typo | Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md) |
| New procedure preset | Acknowledged in paper and README |
| Feature module (new loss, metric, clinical handler) | Co-author on paper |
| Clinical validation data | Co-author on paper |
| Sustained multi-feature contributions | Co-author on paper |

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the current list.

---

## Community Guidelines

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). The short version: be respectful, be constructive, assume good intent.

If you experience or witness unacceptable behavior, report it to the project maintainers. All reports are taken seriously and handled confidentially.
