# landmarkdiff.conditioning

ControlNet conditioning image generation. Produces wireframe renderings and edge maps from facial landmarks for use as conditioning inputs to the diffusion model.

Uses a pre-defined anatomical adjacency matrix (not dynamic Delaunay triangulation) to prevent triangle inversion on drastic landmark displacements. The topology is invariant to deformation.

## Functions

### `render_wireframe`

```python
def render_wireframe(
    face: FaceLandmarks,
    width: int | None = None,
    height: int | None = None,
    thickness: int = 1,
) -> np.ndarray
```

Render the static anatomical adjacency wireframe on a black canvas. Draws contour lines along the jawline, eyes, eyebrows, nose bridge, nose tip, and lips.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face` | `FaceLandmarks` | (required) | Facial landmarks (normalized coordinates) |
| `width` | `int \| None` | `None` | Canvas width (defaults to `face.image_width`) |
| `height` | `int \| None` | `None` | Canvas height (defaults to `face.image_height`) |
| `thickness` | `int` | `1` | Line thickness in pixels |

**Returns:** Grayscale wireframe image (`np.ndarray`, dtype `uint8`).

**Example:**

```python
from landmarkdiff.conditioning import render_wireframe

wireframe = render_wireframe(deformed_face, 512, 512)
cv2.imwrite("wireframe.png", wireframe)
```

---

### `auto_canny`

```python
def auto_canny(image: np.ndarray) -> np.ndarray
```

Compute a Canny edge map with automatic threshold selection adapted to skin tone. Thresholds are derived from the image median rather than hardcoded values, so this works across all Fitzpatrick skin types (I through VI).

Threshold formula:
- Low threshold: `0.66 * median`
- High threshold: `1.33 * median`

After edge detection, applies morphological skeletonization to produce guaranteed 1-pixel-wide edges (ControlNet performs better with thin edges).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` | Grayscale input image (uint8) |

**Returns:** Binary edge map (`np.ndarray`, dtype `uint8`, values 0 or 255).

**Example:**

```python
from landmarkdiff.conditioning import auto_canny

edges = auto_canny(grayscale_image)
```

---

### `generate_conditioning`

```python
def generate_conditioning(
    face: FaceLandmarks,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Generate all conditioning signals for the pipeline in one call. Produces three outputs used at different stages:

1. **Landmark image**: full 2556-edge tessellation mesh (BGR, from `render_landmark_image`). This is the primary ControlNet conditioning signal.
2. **Canny edges**: auto-thresholded edge map of the wireframe (grayscale). Used in compositing.
3. **Wireframe**: anatomical adjacency wireframe (grayscale). Used for visualization and additional conditioning.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face` | `FaceLandmarks` | (required) | Facial landmarks (typically the deformed/manipulated landmarks) |
| `width` | `int \| None` | `None` | Output width (defaults to `face.image_width`) |
| `height` | `int \| None` | `None` | Output height (defaults to `face.image_height`) |

**Returns:** Tuple of `(landmark_image, canny_edges, wireframe)`.

| Return element | Type | Shape | Description |
|----------------|------|-------|-------------|
| `landmark_image` | `np.ndarray` | `(H, W, 3)` | BGR tessellation mesh image |
| `canny_edges` | `np.ndarray` | `(H, W)` | Binary edge map (uint8) |
| `wireframe` | `np.ndarray` | `(H, W)` | Grayscale wireframe (uint8) |

**Example:**

```python
from landmarkdiff.conditioning import generate_conditioning

landmark_img, canny, wireframe = generate_conditioning(manipulated_face, 512, 512)
cv2.imwrite("conditioning_mesh.png", landmark_img)
cv2.imwrite("conditioning_canny.png", canny)
cv2.imwrite("conditioning_wireframe.png", wireframe)
```

---

## Contour Constants

The module defines static contour index lists for each anatomical region:

| Constant | Description | Edge count |
|----------|-------------|------------|
| `JAWLINE_CONTOUR` | Full jawline loop (chin to temples) | 36 |
| `LEFT_EYE_CONTOUR` | Left eye outline (closed loop) | 16 |
| `RIGHT_EYE_CONTOUR` | Right eye outline (closed loop) | 16 |
| `LEFT_EYEBROW` | Left eyebrow arch | 9 |
| `RIGHT_EYEBROW` | Right eyebrow arch | 9 |
| `NOSE_BRIDGE` | Nasal bridge (forehead to tip) | 6 |
| `NOSE_TIP` | Nose tip and columella | 12 |
| `NOSE_BOTTOM` | Base of nose / nostrils | 11 |
| `OUTER_LIPS` | Outer lip vermilion (closed loop) | 21 |
| `INNER_LIPS` | Inner lip line (closed loop) | 20 |
| `ALL_CONTOURS` | List of all contour arrays | -- |

These are used instead of Delaunay triangulation because the topology is fixed and invariant to landmark displacement. Dynamic triangulation can produce inverted triangles when landmarks move significantly.

---

## See Also

- [landmarks](landmarks.md): `FaceLandmarks`, `render_landmark_image`
- [manipulation](manipulation.md): deform landmarks before conditioning
- [inference](inference.md): full pipeline that consumes conditioning images
