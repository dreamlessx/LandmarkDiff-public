# landmarkdiff.inference

Full prediction pipeline combining landmark extraction, Gaussian RBF deformation, ControlNet generation, and Laplacian pyramid compositing.

## Classes

### `LandmarkDiffPipeline`

```python
class LandmarkDiffPipeline:
    def __init__(
        self,
        mode: str = "img2img",
        controlnet_id: str = "CrucibleAI/ControlNetMediaPipeFace",
        controlnet_checkpoint: str | None = None,
        base_model_id: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ip_adapter_scale: float = 0.6,
        clinical_flags: ClinicalFlags | None = None,
        displacement_model_path: str | None = None,
        tps_backend: str = "numpy",
        tps_onnx_path: str | None = None,
    )
```

Main inference pipeline. Supports four modes with different quality-speed-hardware tradeoffs.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"img2img"` | Inference mode: `"tps"`, `"img2img"`, `"controlnet"`, or `"controlnet_ip"` |
| `controlnet_id` | `str` | `"CrucibleAI/ControlNetMediaPipeFace"` | HuggingFace model ID for the ControlNet |
| `controlnet_checkpoint` | `str \| None` | `None` | Path to a fine-tuned ControlNet checkpoint directory |
| `base_model_id` | `str \| None` | `None` | Base SD model ID (defaults to `runwayml/stable-diffusion-v1-5`) |
| `device` | `torch.device \| None` | `None` | Compute device. Auto-detected if not set (CUDA > MPS > CPU). |
| `dtype` | `torch.dtype \| None` | `None` | Model precision. Defaults to `float16` on CUDA, `float32` on MPS/CPU. |
| `ip_adapter_scale` | `float` | `0.6` | IP-Adapter conditioning strength for `controlnet_ip` mode |
| `clinical_flags` | `ClinicalFlags \| None` | `None` | Default clinical flags applied to all predictions |
| `displacement_model_path` | `str \| None` | `None` | Path to a fitted `DisplacementModel` (`.npz`) for data-driven manipulation |
| `tps_backend` | `str` | `"numpy"` | TPS warp backend: `"numpy"` (default) or `"onnx"` |
| `tps_onnx_path` | `str \| None` | `None` | Path to exported TPS ONNX model, used when `tps_backend="onnx"` |

When `tps_backend="onnx"`, the pipeline uses ONNX Runtime for the TPS warp stage and falls back to NumPy/OpenCV TPS if initialization or execution fails.

**Inference Modes:**

| Mode | GPU | Speed | Quality | Identity |
|------|-----|-------|---------|----------|
| `tps` | No | ~0.5s | Geometric only | Perfect |
| `img2img` | Yes (6 GB) | ~5s | Good | Good |
| `controlnet` | Yes (6 GB) | ~5s | Best | Good |
| `controlnet_ip` | Yes (8 GB) | ~7s | Best | Best |

---

#### `load() -> None`

Load model weights into memory. Must be called before `generate()`. For `tps` mode this is a no-op.

```python
pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()
```

---

#### `is_loaded -> bool`

Read-only property. Returns `True` if the pipeline is ready for inference.

---

#### `generate`

```python
def generate(
    self,
    image: np.ndarray,
    procedure: str = "rhinoplasty",
    intensity: float = 50.0,
    num_inference_steps: int = 30,
    guidance_scale: float = 9.0,
    controlnet_conditioning_scale: float = 0.9,
    strength: float = 0.5,
    seed: int | None = None,
    clinical_flags: ClinicalFlags | None = None,
    postprocess: bool = True,
    use_gfpgan: bool = False,
) -> dict
```

Generate a surgical outcome prediction from an input face image. The image is resized to 512x512 internally.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | (required) | Input face photo as BGR numpy array (any resolution; resized to 512x512) |
| `procedure` | `str` | `"rhinoplasty"` | Surgical procedure name |
| `intensity` | `float` | `50.0` | Deformation strength, 0 to 100 |
| `num_inference_steps` | `int` | `30` | Diffusion denoising steps. More steps = higher quality but slower. |
| `guidance_scale` | `float` | `9.0` | Classifier-free guidance strength |
| `controlnet_conditioning_scale` | `float` | `0.9` | How strongly the wireframe controls generation. Keep below 1.2 to avoid saturation. |
| `strength` | `float` | `0.5` | img2img denoising strength (used in img2img mode) |
| `seed` | `int \| None` | `None` | Random seed for reproducible results |
| `clinical_flags` | `ClinicalFlags \| None` | `None` | Per-call clinical flags (overrides constructor default) |
| `postprocess` | `bool` | `True` | Run the post-processing pipeline (CodeFormer, skin tone matching, Laplacian blend, identity check) |
| `use_gfpgan` | `bool` | `False` | Use GFPGAN instead of CodeFormer for face restoration |

**Returns:** `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `"output"` | `np.ndarray` | Final composited result image (BGR, 512x512) |
| `"output_raw"` | `np.ndarray` | Raw diffusion output before compositing |
| `"output_tps"` | `np.ndarray` | TPS-only geometric warp (always computed as baseline) |
| `"input"` | `np.ndarray` | Resized input image (512x512) |
| `"landmarks_original"` | `FaceLandmarks` | Extracted input landmarks |
| `"landmarks_manipulated"` | `FaceLandmarks` | Deformed landmarks |
| `"conditioning"` | `np.ndarray` | Wireframe image fed to ControlNet |
| `"mask"` | `np.ndarray` | Surgical region mask |
| `"procedure"` | `str` | Procedure name used |
| `"intensity"` | `float` | Intensity value used |
| `"device"` | `str` | Device string |
| `"mode"` | `str` | Inference mode used |
| `"view_info"` | `dict` | Face orientation info (yaw, pitch, view classification) |
| `"ip_adapter_active"` | `bool` | Whether IP-Adapter was used |
| `"identity_check"` | `dict \| None` | ArcFace similarity score (if postprocessing enabled) |
| `"restore_used"` | `str` | Face restoration method used |
| `"manipulation_mode"` | `str` | `"preset"` or `"displacement_model"` |

**Raises:**
- `RuntimeError` if `load()` has not been called.
- `ValueError` if no face is detected in the input image.

**Example:**

```python
import cv2
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()

image = cv2.imread("patient.jpg")
result = pipeline.generate(
    image,
    procedure="rhinoplasty",
    intensity=60,
    seed=42,
)

cv2.imwrite("prediction.png", result["output"])
print(f"Identity score: {result['identity_check']}")
print(f"View: {result['view_info']['view']} (yaw={result['view_info']['yaw']})")
```

---

#### `mask_composite`

```python
def mask_composite(
    warped: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    use_laplacian: bool = True,
) -> np.ndarray
```

Composite a generated face image onto the original using a feathered surgical mask. Uses Laplacian pyramid blending by default for seamless transitions. Matches skin tone in LAB color space to prevent color shifts.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warped` | `np.ndarray` | (required) | Generated/warped face image (BGR) |
| `original` | `np.ndarray` | (required) | Original input image (BGR) |
| `mask` | `np.ndarray` | (required) | Surgical mask (float32 [0-1] or uint8 [0-255]) |
| `use_laplacian` | `bool` | `True` | Use 6-level Laplacian pyramid blending |

**Returns:** Composited BGR image (`np.ndarray`).

---

## Standalone Functions

### `estimate_face_view`

```python
def estimate_face_view(face: FaceLandmarks) -> dict
```

Estimate face orientation from landmarks for multi-view awareness. Uses nose tip (idx 1), ear landmarks (idx 234, 454), forehead (idx 10), and chin (idx 152).

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"yaw"` | `float` | Horizontal rotation in degrees (negative = left, positive = right) |
| `"pitch"` | `float` | Vertical rotation in degrees |
| `"view"` | `str` | Classification: `"frontal"`, `"three_quarter"`, or `"profile"` |
| `"is_frontal"` | `bool` | `True` if absolute yaw < 15 degrees |
| `"warning"` | `str \| None` | Warning message if yaw > 30 degrees |

---

### `get_device`

```python
def get_device() -> torch.device
```

Auto-detect the best available compute device. Checks MPS first (Apple Silicon), then CUDA, then falls back to CPU.

---

## See Also

- [landmarks](landmarks.md): landmark extraction and `FaceLandmarks`
- [manipulation](manipulation.md): procedure preset deformations
- [conditioning](conditioning.md): conditioning image generation
- [clinical](clinical.md): clinical edge case flags
- [evaluation](evaluation.md): metrics for evaluating predictions
