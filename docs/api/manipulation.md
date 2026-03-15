# landmarkdiff.manipulation

Gaussian RBF landmark deformation for surgical simulation. Applies procedure-specific displacement vectors to the 478-point MediaPipe face mesh.

## Classes

### `DeformationHandle`

```python
@dataclass(frozen=True)
class DeformationHandle:
    landmark_index: int          # MediaPipe index (0-477)
    displacement: np.ndarray     # (2,) or (3,) pixel displacement vector
    influence_radius: float      # Gaussian RBF radius in pixels
```

A single deformation control point. Each handle moves its target landmark by `displacement` pixels and smoothly influences nearby landmarks within `influence_radius` using a Gaussian falloff.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `landmark_index` | `int` | MediaPipe landmark index (0 to 477) |
| `displacement` | `np.ndarray` | Movement vector `[dx, dy]` or `[dx, dy, dz]` in pixels |
| `influence_radius` | `float` | Gaussian RBF sigma in pixels. Larger values produce smoother, wider deformations. |

---

## Constants

### `PROCEDURE_LANDMARKS`

`dict[str, list[int]]` mapping procedure names to their target landmark indices.

| Procedure | Landmark count | Description |
|-----------|---------------|-------------|
| `"rhinoplasty"` | 25 | Nose reshaping (bridge, tip, alar base) |
| `"blepharoplasty"` | 30 | Eyelid surgery (upper/lower lid, canthi) |
| `"rhytidectomy"` | 33 | Facelift (jawline, cheeks, temples) |
| `"orthognathic"` | 47 | Jaw repositioning (mandible, maxilla, chin) |
| `"brow_lift"` | 19 | Brow elevation and forehead smoothing |
| `"mentoplasty"` | 8 | Chin surgery (advancement, contouring) |

### `PROCEDURE_RADIUS`

`dict[str, float]` mapping procedure names to their default Gaussian RBF influence radius (in pixels at 512x512 resolution).

| Procedure | Radius (px) | Rationale |
|-----------|-------------|-----------|
| `"rhinoplasty"` | 30.0 | Moderate: smooth nasal transitions |
| `"blepharoplasty"` | 15.0 | Tight: avoid affecting brow |
| `"rhytidectomy"` | 40.0 | Wide: broad soft tissue mobilization |
| `"orthognathic"` | 35.0 | Wide: large jaw region |
| `"brow_lift"` | 25.0 | Moderate: brow and forehead |
| `"mentoplasty"` | 25.0 | Moderate: chin and lower contour |

---

## Functions

### `gaussian_rbf_deform`

```python
def gaussian_rbf_deform(
    landmarks: np.ndarray,
    handle: DeformationHandle,
) -> np.ndarray
```

Apply a single Gaussian RBF deformation handle to a landmark array. The deformation formula is `delta * exp(-dist^2 / 2r^2)`, where `dist` is the distance from the handle's landmark and `r` is the influence radius.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `landmarks` | `np.ndarray` | Landmark coordinates, shape `(478, 2)` or `(478, 3)` in pixel space |
| `handle` | `DeformationHandle` | Single deformation control handle |

**Returns:** New `np.ndarray` with deformed positions (input is not modified).

```python
import numpy as np
from landmarkdiff.manipulation import DeformationHandle, gaussian_rbf_deform

handle = DeformationHandle(
    landmark_index=4,                        # nose tip
    displacement=np.array([0.0, -5.0]),      # move 5px upward
    influence_radius=30.0,
)
deformed = gaussian_rbf_deform(pixel_landmarks, handle)
```

---

### `apply_procedure_preset`

```python
def apply_procedure_preset(
    face: FaceLandmarks,
    procedure: str,
    intensity: float = 50.0,
    image_size: int = 512,
    clinical_flags: ClinicalFlags | None = None,
    displacement_model_path: str | None = None,
    noise_scale: float = 0.0,
) -> FaceLandmarks
```

Apply a named surgical procedure preset to face landmarks. This is the main entry point for landmark manipulation.

The `intensity` parameter uses a 0-100 scale. Internally, it is divided by 100 (`scale = intensity / 100.0`) to produce a fractional multiplier applied to all displacement vectors. An intensity of 50 gives half the maximum displacement; 100 gives the full calibrated displacement.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face` | `FaceLandmarks` | (required) | Input face landmarks |
| `procedure` | `str` | (required) | One of `"rhinoplasty"`, `"blepharoplasty"`, `"rhytidectomy"`, `"orthognathic"`, `"brow_lift"`, `"mentoplasty"` |
| `intensity` | `float` | `50.0` | Deformation strength on a 0 to 100 scale. Mild ~ 33, moderate ~ 66, aggressive ~ 100. |
| `image_size` | `int` | `512` | Reference image size for displacement scaling. Displacements are calibrated at 512x512. |
| `clinical_flags` | `ClinicalFlags \| None` | `None` | Clinical condition flags (see [clinical](clinical.md)). Enables condition-specific handling: Ehlers-Danlos widens radii by 1.5x, Bell's palsy removes handles on the paralyzed side. |
| `displacement_model_path` | `str \| None` | `None` | Path to a fitted `DisplacementModel` (`.npz`). When provided, uses data-driven displacements from real surgery pairs instead of hand-tuned RBF vectors. |
| `noise_scale` | `float` | `0.0` | Random variation added to data-driven displacements (0 = deterministic). Only used when `displacement_model_path` is set. |

**Returns:** New `FaceLandmarks` with deformed landmark positions. The original `face` is not modified.

**Raises:** `ValueError` if `procedure` is not a recognized procedure name.

**Example:**

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

face = extract_landmarks(image)
deformed = apply_procedure_preset(face, "rhinoplasty", intensity=60)

# With clinical flags
from landmarkdiff.clinical import ClinicalFlags
flags = ClinicalFlags(ehlers_danlos=True)
deformed = apply_procedure_preset(face, "rhinoplasty", intensity=60, clinical_flags=flags)

# With data-driven displacements
deformed = apply_procedure_preset(
    face, "rhinoplasty", intensity=70,
    displacement_model_path="data/hda_displacement_model.npz",
)
```

---

## Procedure Details

Each procedure preset defines specific displacement vectors for different anatomical sub-regions within the targeted area. All displacements are calibrated at 512x512 resolution and scaled by `image_size / 512`.

**Rhinoplasty**: alar base narrowing (nostrils inward), tip refinement (upward rotation), dorsum narrowing (bridge squeeze).

**Blepharoplasty**: upper lid elevation (central lid strongest), medial/lateral corner tapering (reduced displacement), subtle lower lid tightening.

**Rhytidectomy**: jowl lifting (upward + toward ear, strongest effect), submental tightening (upward only), temple/upper face lift (mild).

**Orthognathic**: mandible repositioning (upward), chin projection (upward), lateral jaw narrowing (bilateral symmetric inward pull).

**Brow lift**: brow elevation with lateral-to-medial gradient (lateral brow lifts more), forehead smoothing (subtle upward shift with wider influence radius).

**Mentoplasty**: chin tip advancement (strongest displacement), lower contour follow-through, jaw angle transition (minimal pull for natural blending).

---

## See Also

- [landmarks](landmarks.md): `FaceLandmarks` dataclass and extraction
- [clinical](clinical.md): `ClinicalFlags` for condition-specific behavior
- [conditioning](conditioning.md): render deformed landmarks as ControlNet input
