# Custom Procedure Presets

Define your own surgical procedure by specifying which landmarks to move and how.

## How it works

LandmarkDiff uses **Gaussian RBF (Radial Basis Function) deformation** to move facial landmarks. Each procedure is defined by a list of **control handles** - anchor points on the face that get displaced in a specific direction.

A control handle has three components:
- **anchor_index**: Which MediaPipe landmark to move (0-477)
- **displacement**: Direction and magnitude of movement `[dx, dy, dz]` in pixels
- **radius**: How far the influence spreads to neighboring landmarks

## Step 1: Identify landmarks

The MediaPipe Face Mesh provides 478 landmarks. Key regions:

| Region | Landmark indices |
|--------|-----------------|
| Nose bridge | 6, 197, 195, 5 |
| Nose tip | 4, 1 |
| Nose wings | 48, 278 |
| Left eye | 33, 133, 159, 145 |
| Right eye | 362, 263, 386, 374 |
| Upper lip | 13, 14, 82, 312 |
| Lower lip | 17, 15, 87, 317 |
| Chin | 152, 175, 148, 377 |
| Jawline left | 234, 132, 58 |
| Jawline right | 454, 361, 288 |
| Left brow | 70, 63, 105, 66 |
| Right brow | 300, 293, 334, 296 |

## Step 2: Define your procedure

```python
import numpy as np
from landmarkdiff.manipulation import ControlHandle

# Example: mentoplasty (chin advancement)
mentoplasty_handles = [
    # Move chin tip forward and slightly down
    ControlHandle(anchor_index=152, displacement=np.array([0, 5, -8]), radius=30.0),
    # Move lower chin forward
    ControlHandle(anchor_index=175, displacement=np.array([0, 3, -6]), radius=25.0),
    # Adjust chin contour
    ControlHandle(anchor_index=148, displacement=np.array([0, 2, -4]), radius=20.0),
    ControlHandle(anchor_index=377, displacement=np.array([0, 2, -4]), radius=20.0),
]
```

## Step 3: Register the preset

Add your procedure to `PROCEDURE_PRESETS` in `landmarkdiff/manipulation.py`:

```python
PROCEDURE_PRESETS = {
    "rhinoplasty": [...],
    "blepharoplasty": [...],
    "rhytidectomy": [...],
    "orthognathic": [...],
    "mentoplasty": mentoplasty_handles,  # your new procedure
}
```

## Step 4: Test it

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

landmarks = extract_landmarks("face.jpg")
deformed = apply_procedure_preset(landmarks, "mentoplasty", intensity=0.6)

# Visualize the deformation
from landmarkdiff.conditioning import draw_tessellation
original_mesh = draw_tessellation(landmarks, (512, 512))
deformed_mesh = draw_tessellation(deformed, (512, 512))
```

## Step 5: Add a test

In `tests/test_manipulation.py`:

```python
def test_mentoplasty_preset():
    landmarks = create_dummy_landmarks()
    result = apply_procedure_preset(landmarks, "mentoplasty", intensity=0.5)
    assert result is not None
    # Chin landmarks should have moved
    assert not np.allclose(result.points[152], landmarks.points[152])
```

## Tips

- Start with small displacements (3-8 pixels) and adjust
- Use larger radii (25-40) for smooth, natural-looking deformations
- Use smaller radii (10-15) for localized changes
- Test with multiple face shapes - the same displacement can look different on different faces
- The intensity parameter (0-1) scales all displacements linearly
