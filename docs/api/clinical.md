# landmarkdiff.clinical

Clinical edge case handling for patients with pathological conditions that affect how facial tissue responds to surgery. These flags modify deformation behavior, mask generation, and compositing to produce more realistic predictions for specific patient populations.

## Classes

### `ClinicalFlags`

```python
@dataclass
class ClinicalFlags:
    vitiligo: bool = False
    bells_palsy: bool = False
    bells_palsy_side: str = "left"
    keloid_prone: bool = False
    keloid_regions: list[str] = field(default_factory=list)
    ehlers_danlos: bool = False
```

Dataclass for clinical condition flags. Pass an instance to `apply_procedure_preset()` or `LandmarkDiffPipeline.generate()` to enable condition-specific pipeline behavior.

**Attributes:**

| Attribute | Type | Default | Effect on pipeline |
|-----------|------|---------|-------------------|
| `vitiligo` | `bool` | `False` | Reduces mask intensity over depigmented patches (preservation factor 0.3) to avoid blending over vitiligo regions |
| `bells_palsy` | `bool` | `False` | Disables deformation handles on the paralyzed side |
| `bells_palsy_side` | `str` | `"left"` | Which side is affected: `"left"` or `"right"` |
| `keloid_prone` | `bool` | `False` | Softens mask transitions in keloid-prone regions (reduction factor 0.5, Gaussian blur sigma 10.0) |
| `keloid_regions` | `list[str]` | `[]` | Anatomical region names prone to keloids, e.g. `["jawline", "nose"]`. Uses names from `LANDMARK_REGIONS`. |
| `ehlers_danlos` | `bool` | `False` | Multiplies Gaussian RBF influence radius by 1.5 for wider, more gradual deformations |

**Methods:**

#### `has_any() -> bool`

Returns `True` if any clinical flag is set.

**Example:**

```python
from landmarkdiff.clinical import ClinicalFlags

flags = ClinicalFlags(
    vitiligo=True,
    bells_palsy=True,
    bells_palsy_side="left",
    keloid_prone=True,
    keloid_regions=["jawline", "nose"],
    ehlers_danlos=False,
)

result = pipeline.generate(
    image,
    procedure="rhinoplasty",
    intensity=60,
    clinical_flags=flags,
)
```

---

## Functions

### `detect_vitiligo_patches`

```python
def detect_vitiligo_patches(
    image: np.ndarray,
    face: FaceLandmarks,
    l_threshold: float = 85.0,
    min_patch_area: int = 200,
) -> np.ndarray
```

Detect depigmented (vitiligo) skin patches using LAB color space analysis. Identifies regions that are significantly brighter than surrounding skin with low color saturation.

Detection criteria:
1. Luminance above threshold (or > 2 standard deviations above face mean)
2. Low saturation in LAB a/b channels (within 15 units of neutral 128)
3. Minimum contour area of `min_patch_area` pixels (filters noise)
4. Restricted to face ROI (convex hull of landmarks)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | (required) | BGR face image |
| `face` | `FaceLandmarks` | (required) | Extracted landmarks (used for face ROI) |
| `l_threshold` | `float` | `85.0` | Luminance threshold for patch detection |
| `min_patch_area` | `int` | `200` | Minimum contour area in pixels to qualify as a patch |

**Returns:** Binary mask (`np.ndarray`, dtype `uint8`, values 0 or 255) of detected vitiligo patches.

---

### `adjust_mask_for_vitiligo`

```python
def adjust_mask_for_vitiligo(
    mask: np.ndarray,
    vitiligo_patches: np.ndarray,
    preservation_factor: float = 0.3,
) -> np.ndarray
```

Reduce surgical mask intensity over vitiligo-affected regions. This preserves depigmented patches in the output rather than blending new skin texture over them.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask` | `np.ndarray` | (required) | Float32 surgical mask [0-1] |
| `vitiligo_patches` | `np.ndarray` | (required) | Binary vitiligo mask (0/255 uint8, from `detect_vitiligo_patches`) |
| `preservation_factor` | `float` | `0.3` | How much to reduce blending. 0 = full blend (no preservation), 1 = fully preserve original. |

**Returns:** Modified float32 mask with reduced intensity over vitiligo patches.

---

### `get_bells_palsy_side_indices`

```python
def get_bells_palsy_side_indices(side: str) -> dict[str, list[int]]
```

Get MediaPipe landmark indices for the affected (paralyzed) side in Bell's palsy. These indices are excluded from deformation to avoid unrealistic manipulation of paralyzed tissue.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `side` | `str` | Affected side: `"left"` or `"right"` |

**Returns:** Dictionary mapping region names to landmark index lists on the affected side.

| Region key | Description |
|------------|-------------|
| `"eye"` | Eye contour landmarks (16 indices) |
| `"eyebrow"` | Eyebrow landmarks (10 indices) |
| `"mouth_corner"` | Mouth corner landmarks (5 indices) |
| `"jawline"` | Jawline landmarks (8 indices) |

---

### `get_keloid_exclusion_mask`

```python
def get_keloid_exclusion_mask(
    face: FaceLandmarks,
    regions: list[str],
    width: int,
    height: int,
    margin_px: int = 10,
) -> np.ndarray
```

Generate a mask of keloid-prone regions. The mask is built from the convex hull of landmarks in each specified region, dilated by a margin.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face` | `FaceLandmarks` | (required) | Extracted landmarks |
| `regions` | `list[str]` | (required) | Region names from `LANDMARK_REGIONS`, e.g. `["jawline", "nose"]` |
| `width` | `int` | (required) | Image width |
| `height` | `int` | (required) | Image height |
| `margin_px` | `int` | `10` | Dilation margin around keloid regions in pixels |

**Returns:** Float32 mask [0-1] where 1.0 indicates keloid-prone area.

---

### `adjust_mask_for_keloid`

```python
def adjust_mask_for_keloid(
    mask: np.ndarray,
    keloid_mask: np.ndarray,
    reduction_factor: float = 0.5,
) -> np.ndarray
```

Soften mask transitions in keloid-prone areas. Reduces mask intensity and applies additional Gaussian blur (kernel 31, sigma 10.0) within keloid regions to prevent sharp compositing boundaries.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask` | `np.ndarray` | (required) | Float32 surgical mask [0-1] |
| `keloid_mask` | `np.ndarray` | (required) | Float32 keloid region mask [0-1] (from `get_keloid_exclusion_mask`) |
| `reduction_factor` | `float` | `0.5` | How much to reduce mask intensity in keloid areas |

**Returns:** Modified float32 mask with gentler transitions in keloid regions.

---

## How Clinical Flags Affect the Pipeline

### Vitiligo

When `vitiligo=True`:
1. `detect_vitiligo_patches()` identifies depigmented regions in LAB color space
2. `adjust_mask_for_vitiligo()` reduces the surgical mask intensity over those patches
3. The compositing step preserves the original vitiligo pattern instead of overwriting it

### Bell's Palsy

When `bells_palsy=True`:
1. `get_bells_palsy_side_indices()` identifies landmarks on the paralyzed side
2. `apply_procedure_preset()` removes all deformation handles targeting those landmarks
3. Only the healthy side is deformed, matching how real surgery would be planned

### Keloid-Prone Skin

When `keloid_prone=True`:
1. `get_keloid_exclusion_mask()` creates a mask of prone regions from `keloid_regions`
2. `adjust_mask_for_keloid()` reduces mask intensity by 0.5 and adds Gaussian blur
3. Compositing produces softer transitions that avoid simulating visible incision lines

### Ehlers-Danlos Syndrome

When `ehlers_danlos=True`:
1. `apply_procedure_preset()` multiplies the Gaussian RBF influence radius by 1.5
2. Deformations spread more widely and gradually
3. This models how hypermobile tissue stretches further than typical tissue during surgery

---

## See Also

- [manipulation](manipulation.md): `apply_procedure_preset` consumes clinical flags
- [inference](inference.md): `LandmarkDiffPipeline.generate()` accepts clinical flags
- [landmarks](landmarks.md): `LANDMARK_REGIONS` for valid region names
