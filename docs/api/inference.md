# landmarkdiff.inference

Full prediction pipeline combining landmark deformation, ControlNet generation, and compositing.

## Classes

### `LandmarkDiffPipeline`

Main inference pipeline.

**Class Methods:**
- `from_pretrained(checkpoint_dir, device="cuda") -> LandmarkDiffPipeline`

**Methods:**

#### `generate(image, procedure, intensity=1.0, mode="controlnet", **kwargs) -> dict`

Generate a surgical prediction.

**Parameters:**
- `image` (str | Path | np.ndarray | PIL.Image): Input face photo
- `procedure` (str): Surgical procedure name
- `intensity` (float): Deformation strength (0.0-1.0)
- `mode` (str): Generation mode
  - `"controlnet"` - full diffusion pipeline (best quality)
  - `"img2img"` - image-to-image translation
  - `"tps"` - thin-plate spline only (CPU, fastest)
- `num_inference_steps` (int): Diffusion steps (default: 30)
- `guidance_scale` (float): Classifier-free guidance (default: 7.5)
- `seed` (int | None): Random seed for reproducibility

**Returns:** dict with keys:
- `"prediction"` (PIL.Image): Predicted post-operative face
- `"comparison"` (PIL.Image): Side-by-side before/after
- `"landmarks_original"` (FaceLandmarks): Input landmarks
- `"landmarks_deformed"` (FaceLandmarks): Deformed landmarks
- `"mesh_original"` (PIL.Image): Original tessellation mesh
- `"mesh_deformed"` (PIL.Image): Deformed tessellation mesh
- `"view"` (dict): Face view angle info

**Example:**
```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline.from_pretrained("checkpoints/latest")
result = pipeline.generate(
    "patient_photo.jpg",
    procedure="rhinoplasty",
    intensity=0.6,
    mode="controlnet",
    seed=42
)
result["prediction"].save("prediction.png")
result["comparison"].save("comparison.png")
```

#### `mask_composite(original, generated, mask) -> PIL.Image`

Composite generated face onto original using feathered mask with LAB skin tone matching.
