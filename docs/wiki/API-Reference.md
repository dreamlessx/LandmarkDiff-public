# API Reference

## Core Pipeline

### `LandmarkDiffPipeline`

**Module:** `landmarkdiff.inference`

The main end-to-end pipeline. Handles image input, landmark extraction, manipulation, generation, and post-processing.

```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipe = LandmarkDiffPipeline(
    mode="controlnet",                              # "tps", "img2img", "controlnet", "controlnet_ip"
    controlnet_id="CrucibleAI/ControlNetMediaPipeFace",
    controlnet_checkpoint=None,                     # path to fine-tuned checkpoint
    base_model_id=None,                             # defaults to runwayml/stable-diffusion-v1-5
    device=None,                                    # auto-detects cuda > mps > cpu
    dtype=None,                                     # fp16 on cuda, fp32 on mps/cpu
    ip_adapter_scale=0.6,                           # IP-Adapter conditioning strength
    clinical_flags=None,                            # ClinicalFlags instance
    displacement_model_path=None,                   # path to .npz displacement model
)
```

#### `load()`
Loads model weights. Must be called before `generate()`. For TPS mode, this is a no-op.

#### `generate(image, ...) -> dict`

Runs the full pipeline on a single image.

```python
result = pipe.generate(
    image,                              # np.ndarray BGR (any size, resized to 512x512)
    procedure="rhinoplasty",            # procedure name
    intensity=50.0,                     # 0-100 scale
    num_inference_steps=30,             # diffusion steps (ignored in TPS mode)
    guidance_scale=9.0,                 # classifier-free guidance
    controlnet_conditioning_scale=0.9,  # ControlNet conditioning strength
    strength=0.5,                       # img2img denoising strength
    seed=None,                          # reproducibility seed
    clinical_flags=None,                # overrides pipeline-level flags
    postprocess=True,                   # run neural + classical postprocessing
    use_gfpgan=False,                   # use GFPGAN/CodeFormer face restoration
)
```

**Returns** a dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `output` | np.ndarray | Final composited result (512x512 BGR) |
| `output_raw` | np.ndarray | Raw diffusion/warp output before compositing |
| `output_tps` | np.ndarray | TPS warp result (always computed) |
| `input` | np.ndarray | Resized input (512x512) |
| `landmarks_original` | FaceLandmarks | Detected landmarks |
| `landmarks_manipulated` | FaceLandmarks | Deformed landmarks |
| `conditioning` | np.ndarray | ControlNet conditioning image (face mesh) |
| `mask` | np.ndarray | Surgical mask (float32, 0-1) |
| `procedure` | str | Procedure name |
| `intensity` | float | Intensity used |
| `device` | str | Device string |
| `mode` | str | Inference mode used |
| `view_info` | dict | Face orientation (yaw, pitch, view classification) |
| `ip_adapter_active` | bool | Whether IP-Adapter was loaded |
| `identity_check` | dict or None | ArcFace identity verification result |
| `restore_used` | str | "codeformer", "gfpgan", or "none" |
| `manipulation_mode` | str | "preset" or "displacement_model" |

---

## Landmark Extraction

### `FaceLandmarks`

**Module:** `landmarkdiff.landmarks`

Frozen dataclass holding extracted facial landmarks.

```python
@dataclass(frozen=True)
class FaceLandmarks:
    landmarks: np.ndarray    # (478, 3) normalized (x, y, z) in [0, 1]
    image_width: int
    image_height: int
    confidence: float
```

#### `pixel_coords` (property)

**Important:** This is a `@property`, not a method. Do not call it with parentheses as a function.

```python
# Correct:
coords = face.pixel_coords    # returns (478, 2) array

# Wrong:
coords = face.pixel_coords()  # This also works but is redundant
```

Returns `(478, 2)` array of (x, y) pixel coordinates by multiplying normalized coordinates by image dimensions.

#### `get_region(region: str) -> np.ndarray`

Returns the normalized landmarks for a named anatomical region.

```python
nose_landmarks = face.get_region("nose")  # returns (N, 3) array
```

Valid region names: `"jawline"`, `"eye_left"`, `"eye_right"`, `"eyebrow_left"`, `"eyebrow_right"`, `"nose"`, `"lips"`, `"iris_left"`, `"iris_right"`.

### `extract_landmarks(image, ...) -> FaceLandmarks | None`

```python
from landmarkdiff.landmarks import extract_landmarks

face = extract_landmarks(
    image,                          # BGR numpy array
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

Returns `None` if no face is detected.

---

## Deformation

### `DeformationHandle`

**Module:** `landmarkdiff.manipulation`

```python
@dataclass(frozen=True)
class DeformationHandle:
    landmark_index: int          # which landmark is the center of deformation
    displacement: np.ndarray     # (2,) or (3,) pixel displacement vector
    influence_radius: float      # Gaussian RBF radius in pixels
```

### `gaussian_rbf_deform(landmarks, handle) -> np.ndarray`

Applies a single Gaussian RBF deformation to all landmarks.

```python
from landmarkdiff.manipulation import gaussian_rbf_deform, DeformationHandle

handle = DeformationHandle(
    landmark_index=1,
    displacement=np.array([0.0, -5.0]),
    influence_radius=30.0,
)
new_landmarks = gaussian_rbf_deform(landmarks_array, handle)
```

### `apply_procedure_preset(face, procedure, ...) -> FaceLandmarks`

Applies a full surgical procedure preset.

```python
from landmarkdiff.manipulation import apply_procedure_preset

deformed = apply_procedure_preset(
    face,                          # FaceLandmarks
    "rhinoplasty",                 # procedure name
    intensity=65.0,                # 0-100 scale
    image_size=512,                # reference size for displacement scaling
    clinical_flags=None,           # ClinicalFlags instance
    displacement_model_path=None,  # path to fitted DisplacementModel
    noise_scale=0.0,               # variation noise for data-driven mode
)
```

---

## Conditioning

### `render_wireframe(face, ...) -> np.ndarray`

**Module:** `landmarkdiff.conditioning`

Renders a static anatomical adjacency wireframe on black canvas. Uses predefined contour connections (jawline, eyes, eyebrows, nose, lips) rather than dynamic Delaunay triangulation, which prevents triangle inversion on drastic displacements.

```python
from landmarkdiff.conditioning import render_wireframe
wireframe = render_wireframe(face, width=512, height=512, thickness=1)
```

### `auto_canny(image) -> np.ndarray`

Adaptive Canny edge detection with median-based thresholds (0.66 * median, 1.33 * median). Works across all Fitzpatrick skin types. Post-processes with morphological skeletonization for 1-pixel edges.

### `generate_conditioning(face, ...) -> tuple`

Generates the full ControlNet conditioning signal as a 3-tuple:

```python
from landmarkdiff.conditioning import generate_conditioning
landmark_img, canny_edges, wireframe = generate_conditioning(face, 512, 512)
```

---

## Landmark Visualization

### `render_landmark_image(face, ...) -> np.ndarray`

**Module:** `landmarkdiff.landmarks`

Renders the full MediaPipe tessellation mesh (2556 edges) on black canvas. This matches what CrucibleAI's ControlNet was trained on. Falls back to colored dots if tessellation connections are unavailable.

### `visualize_landmarks(image, face, ...) -> np.ndarray`

Draws colored landmark dots on the image by anatomical region.

---

## Displacement Model

### `DisplacementModel`

**Module:** `landmarkdiff.displacement_model`

Statistical model of per-procedure surgical displacements, fitted from real before/after surgery pairs.

```python
from landmarkdiff.displacement_model import DisplacementModel

# Fit from extracted data
model = DisplacementModel()
model.fit(displacement_list)  # list of dicts from extract_displacements()
model.save("model.npz")

# Load pre-fitted model
model = DisplacementModel.load("model.npz")

# Generate displacement field
field = model.get_displacement_field(
    procedure="rhinoplasty",
    intensity=1.0,         # 1.0 = average observed displacement
    noise_scale=0.3,       # stochastic variation
    rng=np.random.default_rng(42),
)
# field is (478, 2) in normalized coordinate space
```

**Properties:**
- `procedures`: list of procedure names the model was fitted on
- `fitted`: whether the model has been fitted
- `stats`: nested dict of per-procedure statistics
- `n_samples`: dict of sample counts per procedure

### `extract_displacements(before_img, after_img) -> dict | None`

Extracts landmark displacements from a single before/after surgery pair.

### `extract_from_directory(pairs_dir, ...) -> list[dict]`

Batch extraction from a directory. Supports naming conventions: `<name>_before.*` / `<name>_after.*` or `<name>_input.*` / `<name>_target.*`.

---

## Clinical Flags

### `ClinicalFlags`

**Module:** `landmarkdiff.clinical`

```python
from landmarkdiff.clinical import ClinicalFlags

flags = ClinicalFlags(
    vitiligo=False,
    bells_palsy=False,
    bells_palsy_side="left",       # "left" or "right"
    keloid_prone=False,
    keloid_regions=[],             # e.g. ["jawline", "nose"]
    ehlers_danlos=False,
)
```

See the [Clinical Flags](Clinical-Flags) page for detailed behavior.

---

## Post-Processing

### `full_postprocess(generated, original, mask, ...) -> dict`

**Module:** `landmarkdiff.postprocess`

Runs the complete neural + classical post-processing pipeline.

```python
from landmarkdiff.postprocess import full_postprocess

result = full_postprocess(
    generated=raw_output,
    original=original_image,
    mask=surgical_mask,
    restore_mode="codeformer",    # "codeformer", "gfpgan", or "none"
    codeformer_fidelity=0.7,
    use_realesrgan=True,
    use_laplacian_blend=True,
    sharpen_strength=0.25,
    verify_identity=True,
    identity_threshold=0.6,
)
# result["image"], result["identity_check"], result["restore_used"]
```

### `laplacian_pyramid_blend(source, target, mask, levels=6) -> np.ndarray`

Multi-band Laplacian pyramid blending for seamless compositing.

### `verify_identity_arcface(original, result, threshold=0.6) -> dict`

Computes ArcFace cosine similarity between original and output faces.
