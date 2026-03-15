# landmarkdiff.landmarks

Face mesh extraction using MediaPipe Face Mesh v2 (478 3D landmarks).

## Classes

### `FaceLandmarks`

```python
@dataclass(frozen=True)
class FaceLandmarks:
    landmarks: np.ndarray       # (478, 3) normalized x, y, z in [0, 1]
    image_width: int
    image_height: int
    confidence: float
```

Immutable dataclass holding 478-point facial landmark data extracted from a single face.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `landmarks` | `np.ndarray` | Shape `(478, 3)` array of (x, y, z) coordinates normalized to [0, 1] |
| `image_width` | `int` | Width of the source image in pixels |
| `image_height` | `int` | Height of the source image in pixels |
| `confidence` | `float` | Detection confidence score (1.0 if face detected successfully) |

**Properties:**

#### `pixel_coords -> np.ndarray`

Read-only property (not a method) that converts normalized landmarks to pixel coordinates.

Returns a `(478, 2)` array where column 0 is the x-coordinate scaled by `image_width` and column 1 is the y-coordinate scaled by `image_height`. The z-coordinate is dropped.

```python
# pixel_coords is a @property, access it without parentheses:
coords = face.pixel_coords       # correct
# coords = face.pixel_coords()  # WRONG - this will raise TypeError
print(coords.shape)              # (478, 2)
```

**Methods:**

#### `get_region(region: str) -> np.ndarray`

Return the normalized landmark coordinates for a named anatomical region.

**Parameters:**
- `region` (str): Region name from `LANDMARK_REGIONS` (see table below).

**Returns:** `np.ndarray` of shape `(N, 3)` containing the normalized (x, y, z) coordinates for all landmarks in the region. Returns an empty array if the region name is not recognized.

```python
nose = face.get_region("nose")
print(nose.shape)   # (25, 3)
```

---

## Constants

### `LANDMARK_REGIONS`

`dict[str, list[int]]` mapping region names to MediaPipe landmark indices.

| Region | Count | Example indices |
|--------|-------|-----------------|
| `jawline` | 35 | 10, 338, 297, 332, 284, 251, ... |
| `eye_left` | 16 | 33, 7, 163, 144, 145, 153, ... |
| `eye_right` | 16 | 362, 382, 381, 380, 374, 373, ... |
| `eyebrow_left` | 10 | 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 |
| `eyebrow_right` | 10 | 300, 293, 334, 296, 336, 285, 295, 282, 283, 276 |
| `nose` | 25 | 1, 2, 4, 5, 6, 19, 94, 141, ... |
| `lips` | 22 | 61, 146, 91, 181, 84, 17, ... |
| `iris_left` | 5 | 468, 469, 470, 471, 472 |
| `iris_right` | 5 | 473, 474, 475, 476, 477 |

### `REGION_COLORS`

`dict[str, tuple[int, int, int]]` mapping region names to BGR color tuples used for visualization.

---

## Functions

### `extract_landmarks`

```python
def extract_landmarks(
    image: np.ndarray,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> FaceLandmarks | None
```

Extract 478 facial landmarks from an image using MediaPipe Face Mesh. Tries the new MediaPipe Tasks API first (>= 0.10.20), then falls back to the legacy Solutions API.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | (required) | BGR image as a numpy array |
| `min_detection_confidence` | `float` | `0.5` | Minimum face detection confidence threshold |
| `min_tracking_confidence` | `float` | `0.5` | Minimum landmark tracking confidence threshold |

**Returns:** `FaceLandmarks` if a face is detected, `None` otherwise.

**Example:**

```python
from landmarkdiff.landmarks import extract_landmarks
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg").convert("RGB"))[:, :, ::-1]  # RGB to BGR
face = extract_landmarks(img)
if face is not None:
    print(f"Detected {len(face.landmarks)} landmarks")
    print(f"Nose tip (normalized): {face.landmarks[4]}")
    print(f"Nose tip (pixels): {face.pixel_coords[4]}")
```

---

### `visualize_landmarks`

```python
def visualize_landmarks(
    image: np.ndarray,
    face: FaceLandmarks,
    radius: int = 1,
    draw_regions: bool = True,
) -> np.ndarray
```

Draw colored landmark dots on an image, grouped by anatomical region.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | (required) | BGR image to annotate (will be copied, not modified) |
| `face` | `FaceLandmarks` | (required) | Extracted face landmarks |
| `radius` | `int` | `1` | Dot radius in pixels |
| `draw_regions` | `bool` | `True` | Color dots by region. If `False`, all dots are white. |

**Returns:** Annotated BGR image copy (`np.ndarray`).

```python
from landmarkdiff.landmarks import extract_landmarks, visualize_landmarks
annotated = visualize_landmarks(image, face, radius=2)
cv2.imwrite("landmarks.png", annotated)
```

---

### `render_landmark_image`

```python
def render_landmark_image(
    face: FaceLandmarks,
    width: int | None = None,
    height: int | None = None,
    radius: int = 2,
) -> np.ndarray
```

Render the full 2556-edge MediaPipe face mesh tessellation on a black canvas. This is the conditioning image format that CrucibleAI/ControlNetMediaPipeFace was pre-trained on. Falls back to colored dots if the tessellation connections are not available (older MediaPipe versions).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face` | `FaceLandmarks` | (required) | Extracted face landmarks |
| `width` | `int \| None` | `None` | Canvas width (defaults to `face.image_width`) |
| `height` | `int \| None` | `None` | Canvas height (defaults to `face.image_height`) |
| `radius` | `int` | `2` | Dot radius for fallback rendering |

**Returns:** BGR image (`np.ndarray`) with face mesh on black background.

```python
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
mesh_img = render_landmark_image(face, 512, 512)
```

---

### `load_image`

```python
def load_image(path: str | Path) -> np.ndarray
```

Load an image from disk as a BGR numpy array. Raises `FileNotFoundError` if the file cannot be read.

---

## See Also

- [manipulation](manipulation.md): apply surgical deformations to extracted landmarks
- [conditioning](conditioning.md): generate ControlNet conditioning from landmarks
- [inference](inference.md): full prediction pipeline
