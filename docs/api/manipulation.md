# landmarkdiff.manipulation

Gaussian RBF landmark deformation for surgical simulation.

## Classes

### `ControlHandle`

A single deformation control point.

**Attributes:**
- `anchor_index` (int): MediaPipe landmark index (0-477)
- `displacement` (np.ndarray): Movement vector `[dx, dy, dz]` in pixels
- `radius` (float): Gaussian influence radius in pixels

## Constants

### `PROCEDURE_PRESETS`

Pre-defined control handle sets for supported procedures:
- `"rhinoplasty"` - nose reshaping
- `"blepharoplasty"` - eyelid surgery
- `"rhytidectomy"` - facelift
- `"orthognathic"` - jaw surgery

## Functions

### `gaussian_rbf_deform(landmarks, handles, intensity=1.0) -> FaceLandmarks`

Apply Gaussian RBF deformation to landmarks.

**Parameters:**
- `landmarks` (FaceLandmarks): Input landmarks
- `handles` (list[ControlHandle]): Deformation control handles
- `intensity` (float): Deformation strength, 0.0 to 1.0

**Returns:** New `FaceLandmarks` with deformed positions

### `apply_procedure_preset(landmarks, procedure, intensity=1.0) -> FaceLandmarks`

Apply a named procedure preset.

**Parameters:**
- `landmarks` (FaceLandmarks): Input landmarks
- `procedure` (str): One of `"rhinoplasty"`, `"blepharoplasty"`, `"rhytidectomy"`, `"orthognathic"`
- `intensity` (float): Deformation strength, 0.0 to 1.0

**Returns:** New `FaceLandmarks` with deformed positions

**Example:**
```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

landmarks = extract_landmarks("face.jpg")
deformed = apply_procedure_preset(landmarks, "rhinoplasty", intensity=0.6)
```
