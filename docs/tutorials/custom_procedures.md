# Custom Procedure Presets

Define your own surgical procedure by specifying which landmarks to move and how. This tutorial walks through the full process: identifying landmarks, creating the preset, testing it, adding documentation, and submitting a PR.

Two community contributors have already gone through this process successfully:
- **Deepak8858** added the brow_lift procedure (PR [#35](https://github.com/dreamlessx/LandmarkDiff-public/pull/35))
- **P-r-e-m-i-u-m** added the mentoplasty procedure (PR [#36](https://github.com/dreamlessx/LandmarkDiff-public/pull/36))

Both PRs are good references for the scope of changes needed.

## How It Works

LandmarkDiff uses **Gaussian RBF (Radial Basis Function) deformation** to move facial landmarks. Each procedure is defined by a list of **deformation handles**: anchor points on the face that get displaced in a specific direction, with influence that falls off smoothly to neighboring landmarks.

A deformation handle has three components:
- **`landmark_index`**: which MediaPipe landmark to move (0-477)
- **`displacement`**: direction and magnitude of movement `[dx, dy]` in pixels at 512x512 resolution
- **`influence_radius`**: how far the influence spreads to neighboring landmarks (Gaussian falloff)

The deformation formula is: `delta * exp(-dist^2 / 2r^2)`, where `r` is the influence radius.

## Step 1: Research the Procedure

Before writing any code, understand the anatomy:

- Which structures are affected by the surgery? (e.g., for lip augmentation: upper lip vermilion, lower lip vermilion, oral commissures, philtrum)
- In what directions does tissue move? (e.g., lip augmentation pushes the lip border outward/downward for upper lip, outward/upward for lower lip)
- How does the effect vary across the region? (e.g., the center of the lip moves more than the corners)

Look at surgical textbooks, published anthropometric data, or before/after imagery. The review process will check anatomical plausibility.

## Step 2: Identify Landmarks

The MediaPipe Face Mesh provides 478 landmarks. Visualize them on a test face:

```bash
python examples/landmark_visualization.py /path/to/face.jpg
```

Key landmark regions for reference:

| Region | Landmark indices |
|--------|-----------------|
| Nose bridge | 6, 197, 195, 5 |
| Nose tip | 4, 1 |
| Nose wings (alae) | 48, 278, 240, 460 |
| Left eye (upper lid) | 159, 160, 161 |
| Left eye (lower lid) | 145, 153, 154 |
| Right eye (upper lid) | 386, 385, 384 |
| Right eye (lower lid) | 374, 380, 381 |
| Upper lip | 13, 14, 82, 312, 0, 267, 269, 270 |
| Lower lip | 17, 15, 87, 317, 14, 16, 84, 314 |
| Chin | 152, 175, 148, 377 |
| Jawline left | 234, 132, 58 |
| Jawline right | 454, 361, 288 |
| Left brow | 70, 63, 105, 66 |
| Right brow | 300, 293, 334, 296 |
| Forehead | 9, 8, 10 |
| Left ear | 234, 93, 132 |
| Right ear | 454, 323, 361 |

For a complete map, see the [MediaPipe Face Mesh UV visualization](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png).

## Step 3: Create the Preset

You need to modify three things in `landmarkdiff/manipulation.py`:

### a) Add landmark indices to `PROCEDURE_LANDMARKS`

List every landmark your procedure affects:

```python
PROCEDURE_LANDMARKS: dict[str, list[int]] = {
    # ... existing procedures ...
    "lip_augmentation": [
        # Upper lip vermilion
        0, 13, 14, 82, 312, 267, 269, 270,
        # Lower lip vermilion
        17, 15, 87, 317, 14, 16, 84, 314,
        # Oral commissures
        61, 291,
        # Philtrum columns
        164, 167,
    ],
}
```

### b) Set the influence radius in `PROCEDURE_RADIUS`

Choose a radius (in pixels at 512x512) based on the anatomical area:

| Radius range | Use case |
|-------------|----------|
| 10-15 px | Fine structures (eyelids, lip border) |
| 20-25 px | Moderate areas (nose tip, chin, brow) |
| 30-40 px | Broad tissue mobilization (facelift, jawline) |

```python
PROCEDURE_RADIUS: dict[str, float] = {
    # ... existing ...
    "lip_augmentation": 15.0,
}
```

### c) Add displacement vectors in `_get_procedure_handles()`

This is the core of your procedure. Define the displacement direction and magnitude for each landmark or group of landmarks:

```python
def _get_procedure_handles(procedure, indices, scale, radius):
    handles = []
    # ... existing procedure cases ...

    elif procedure == "lip_augmentation":
        # Upper lip: push outward (downward in image coordinates)
        upper_lip_center = [0, 13, 14, 82, 312]
        for idx in upper_lip_center:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 2.5 * scale]),
                        influence_radius=radius,
                    )
                )

        # Upper lip sides: less displacement (tapered)
        upper_lip_sides = [267, 269, 270]
        for idx in upper_lip_sides:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 1.5 * scale]),
                        influence_radius=radius * 0.7,
                    )
                )

        # Lower lip: push outward (upward in image coordinates)
        lower_lip_center = [17, 15, 87, 317]
        for idx in lower_lip_center:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale]),
                        influence_radius=radius,
                    )
                )

        # Philtrum columns: subtle vertical stretch
        philtrum = [164, 167]
        for idx in philtrum:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -0.8 * scale]),
                        influence_radius=radius * 0.5,
                    )
                )

    return handles
```

**Important conventions:**
- Displacements are in pixels at 512x512 resolution. They scale linearly with `scale` (which is `intensity / 100`).
- Positive X is rightward, positive Y is downward.
- Use sub-regions with different magnitudes and radius factors for anatomically realistic results (the center of a region usually moves more than its edges).
- Guard every landmark with `if idx in indices` so the procedure works correctly even if landmark lists are modified later.

## Step 4: Test It

### Quick visual test

```python
import numpy as np
from PIL import Image
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import render_wireframe

img = np.array(Image.open("face.jpg").convert("RGB").resize((512, 512)))
face = extract_landmarks(img)
deformed = apply_procedure_preset(face, "lip_augmentation", intensity=60)

# Visualize the mesh deformation
original_mesh = render_wireframe(face, (512, 512))
deformed_mesh = render_wireframe(deformed, (512, 512))
```

### Run inference

```bash
python scripts/run_inference.py face.jpg --procedure lip_augmentation --mode tps --intensity 50
```

TPS mode (CPU-only) is the fastest way to iterate on displacement vectors since it does not require a GPU or pretrained checkpoint.

### Check multiple intensity levels

```bash
for intensity in 20 40 60 80; do
    python scripts/run_inference.py face.jpg \
        --procedure lip_augmentation \
        --mode tps \
        --intensity $intensity \
        --output output/lip_aug_${intensity}/
done
```

Verify that:
- Low intensity (20) produces a subtle, barely visible change
- Mid intensity (50-60) looks natural and plausible
- High intensity (80+) is exaggerated but not distorted
- The deformation is bilaterally symmetric (unless the procedure is intentionally asymmetric)
- No landmarks cross each other or create mesh inversions

### Test on diverse faces

The same displacement can look different on different face shapes, sizes, and poses. Test with at least 3-5 different face images.

## Step 5: Add a Unit Test

Create or add to `tests/test_manipulation.py`:

```python
import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import (
    PROCEDURE_LANDMARKS,
    PROCEDURE_RADIUS,
    apply_procedure_preset,
)


def create_dummy_landmarks(n: int = 478) -> FaceLandmarks:
    """Create dummy landmarks on a regular grid for testing."""
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.1, 0.9, size=(n, 3))
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=512,
        image_height=512,
        confidence=0.95,
    )


class TestLipAugmentation:
    def test_preset_registered(self):
        """Procedure must be in both PROCEDURE_LANDMARKS and PROCEDURE_RADIUS."""
        assert "lip_augmentation" in PROCEDURE_LANDMARKS
        assert "lip_augmentation" in PROCEDURE_RADIUS

    def test_basic_deformation(self):
        """apply_procedure_preset should return a new FaceLandmarks with moved points."""
        face = create_dummy_landmarks()
        result = apply_procedure_preset(face, "lip_augmentation", intensity=50)
        assert result is not None
        assert result.landmarks.shape == face.landmarks.shape
        # At least some landmarks should have moved
        assert not np.allclose(result.landmarks, face.landmarks)

    def test_zero_intensity(self):
        """Intensity 0 should produce no change."""
        face = create_dummy_landmarks()
        result = apply_procedure_preset(face, "lip_augmentation", intensity=0)
        np.testing.assert_allclose(result.landmarks, face.landmarks, atol=1e-10)

    def test_affected_landmarks_move(self):
        """Procedure-specific landmarks should be displaced."""
        face = create_dummy_landmarks()
        result = apply_procedure_preset(face, "lip_augmentation", intensity=60)
        affected = PROCEDURE_LANDMARKS["lip_augmentation"]
        for idx in affected:
            # At least some affected landmarks should have moved
            if not np.allclose(result.landmarks[idx], face.landmarks[idx], atol=1e-6):
                return  # test passes if any affected landmark moved
        pytest.fail("No affected landmarks were displaced")

    def test_intensity_scaling(self):
        """Higher intensity should produce larger displacements."""
        face = create_dummy_landmarks()
        low = apply_procedure_preset(face, "lip_augmentation", intensity=20)
        high = apply_procedure_preset(face, "lip_augmentation", intensity=80)
        diff_low = np.abs(low.landmarks - face.landmarks).sum()
        diff_high = np.abs(high.landmarks - face.landmarks).sum()
        assert diff_high > diff_low
```

Run your tests:

```bash
pytest tests/test_manipulation.py -v -k lip_augmentation
```

## Step 6: Add Documentation

Create a procedure doc at `docs/procedures/lip_augmentation.md`. Follow the pattern of existing procedures (see `docs/procedures/brow_lift.md` or `docs/procedures/mentoplasty.md`):

```markdown
# Lip Augmentation

Lip volumization targeting the upper and lower lip vermilion borders.

## Anatomy

Lip augmentation modifies three zones:
- **Upper lip vermilion**: ...
- **Lower lip vermilion**: ...
- **Philtrum**: ...

## Affected Landmarks
...

## Displacement Behavior
...

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | ... |
| Moderate | 40-65 | ... |
| Aggressive | 70-100 | ... |

## Code Example

...
```

## Step 7: Run All Checks

Before submitting, make sure everything passes:

```bash
# Lint
ruff check landmarkdiff/ scripts/ tests/

# Type check
mypy landmarkdiff/ --ignore-missing-imports

# Tests
pytest tests/ -v

# Or all at once
make check
```

## Step 8: Submit a Pull Request

### Fork and branch

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/LandmarkDiff-public.git
cd LandmarkDiff-public
git checkout -b add-lip-augmentation-preset
```

### Commit your changes

Your PR should touch these files:

| File | Change |
|------|--------|
| `landmarkdiff/manipulation.py` | Add to `PROCEDURE_LANDMARKS`, `PROCEDURE_RADIUS`, and `_get_procedure_handles()` |
| `tests/test_manipulation.py` | Add tests for the new procedure |
| `docs/procedures/lip_augmentation.md` | Procedure documentation |

```bash
git add landmarkdiff/manipulation.py tests/test_manipulation.py docs/procedures/lip_augmentation.md
git commit -m "Add lip_augmentation procedure preset

Targets 20 upper/lower lip landmarks with outward displacement
for lip volumization. Includes unit tests and procedure docs."
```

### Open the PR

Push your branch and open a pull request against `main`. Fill out the PR template and reference any related issues.

### What reviewers check

- **Anatomical plausibility**: do the displacement directions match real surgical outcomes?
- **Bilateral symmetry**: left and right sides should be mirrors unless the procedure is inherently asymmetric
- **Intensity range**: subtle at low values, exaggerated but not broken at high values
- **Test coverage**: at least 4-5 tests covering registration, deformation, zero intensity, and scaling
- **Documentation**: procedure doc with anatomy, landmarks, displacement table, and intensity recommendations
- **CI passing**: lint, type check, and test suite all green

## Tips

- Start with small displacements (2-4 pixels) and adjust. It is easier to increase magnitude than to fix a distorted mesh.
- Use larger radii (25-40 px) for smooth, natural-looking deformations across broad regions.
- Use smaller radii (10-15 px) for localized, precise changes (eyelids, lip border).
- Test with multiple face shapes. The same displacement can look very different on a narrow face versus a wide face.
- The intensity parameter (0-100) scales all displacements linearly. Design your displacements so that intensity=50 looks "moderate" and intensity=100 is the surgical maximum.
- Study how existing procedures handle sub-regions with different weights. For example, rhinoplasty uses separate groups for the alar base, tip, and dorsum, each with different displacement directions and radius factors.
- The clinical flags system (`bells_palsy`, `ehlers_danlos`) applies automatically to all procedures. If your procedure has special interactions with a clinical condition, document them.

## Reference: Existing Procedures

| Procedure | Landmarks | Radius | Contributed by |
|-----------|-----------|--------|----------------|
| rhinoplasty | 25 | 30 px | core team |
| blepharoplasty | 29 | 15 px | core team |
| rhytidectomy | 30 | 40 px | core team |
| orthognathic | 42 | 35 px | core team |
| brow_lift | 19 | 25 px | Deepak8858 (PR [#35](https://github.com/dreamlessx/LandmarkDiff-public/pull/35)) |
| mentoplasty | 8 | 25 px | P-r-e-m-i-u-m (PR [#36](https://github.com/dreamlessx/LandmarkDiff-public/pull/36)) |
