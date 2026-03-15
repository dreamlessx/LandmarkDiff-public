# landmarkdiff.evaluation

Evaluation metrics for surgical prediction quality. All metrics support stratification by Fitzpatrick skin type (I through VI) and by procedure for equitable performance reporting.

## Classes

### `EvalMetrics`

```python
@dataclass
class EvalMetrics:
    fid: float = 0.0
    lpips: float = 0.0
    nme: float = 0.0
    identity_sim: float = 0.0
    ssim: float = 0.0

    # Per-Fitzpatrick stratification
    fid_by_fitzpatrick: dict[str, float]
    nme_by_fitzpatrick: dict[str, float]
    lpips_by_fitzpatrick: dict[str, float]
    ssim_by_fitzpatrick: dict[str, float]
    identity_sim_by_fitzpatrick: dict[str, float]
    count_by_fitzpatrick: dict[str, int]

    # Per-procedure stratification
    nme_by_procedure: dict[str, float]
    lpips_by_procedure: dict[str, float]
    ssim_by_procedure: dict[str, float]
```

Container for all evaluation metrics with Fitzpatrick skin type and procedure stratification.

**Aggregate Attributes:**

| Attribute | Type | Direction | Target | Description |
|-----------|------|-----------|--------|-------------|
| `fid` | `float` | Lower is better | < 50 | Frechet Inception Distance |
| `lpips` | `float` | Lower is better | < 0.15 | Learned Perceptual Image Patch Similarity |
| `nme` | `float` | Lower is better | < 0.05 | Normalized Mean Error (landmark accuracy) |
| `identity_sim` | `float` | Higher is better | > 0.85 | ArcFace cosine similarity |
| `ssim` | `float` | Higher is better | > 0.80 | Structural Similarity Index |

**Stratification Attributes:**

All `*_by_fitzpatrick` dicts are keyed by Fitzpatrick type string (`"I"` through `"VI"`). All `*_by_procedure` dicts are keyed by procedure name (e.g., `"rhinoplasty"`).

**Methods:**

#### `summary() -> str`

Return a human-readable summary of all metrics, including per-Fitzpatrick breakdowns.

#### `to_dict() -> dict`

Export all metrics as a flat JSON-serializable dictionary. Keys use prefixes like `fitz_III_lpips` and `proc_rhinoplasty_nme`.

```python
metrics = evaluate_batch(predictions, targets)
print(metrics.summary())

import json
with open("eval_results.json", "w") as f:
    json.dump(metrics.to_dict(), f, indent=2)
```

---

## Functions

### `compute_nme`

```python
def compute_nme(
    pred_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    left_eye_idx: int = 33,
    right_eye_idx: int = 263,
) -> float
```

Compute Normalized Mean Error between predicted and target landmarks. The error is normalized by inter-ocular distance (distance between landmarks 33 and 263) for scale invariance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_landmarks` | `np.ndarray` | (required) | Predicted landmark positions, shape `(N, 2)` |
| `target_landmarks` | `np.ndarray` | (required) | Ground truth positions, shape `(N, 2)` |
| `left_eye_idx` | `int` | `33` | MediaPipe index for left eye center |
| `right_eye_idx` | `int` | `263` | MediaPipe index for right eye center |

**Returns:** NME value as `float` (lower is better).

```python
from landmarkdiff.evaluation import compute_nme
nme = compute_nme(pred_face.pixel_coords, target_face.pixel_coords)
```

---

### `compute_ssim`

```python
def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
) -> float
```

Compute Structural Similarity Index (SSIM) between two images. Uses scikit-image's windowed SSIM implementation (Wang et al. 2004) with an 11x11 Gaussian kernel when available. Falls back to a simple global SSIM if scikit-image is not installed.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pred` | `np.ndarray` | Predicted image (BGR or grayscale, uint8) |
| `target` | `np.ndarray` | Target image (same shape as `pred`) |

**Returns:** SSIM score as `float` in range [0, 1] (higher is better).

```python
from landmarkdiff.evaluation import compute_ssim
ssim = compute_ssim(result["output"], target_image)
```

---

### `compute_lpips`

```python
def compute_lpips(
    pred: np.ndarray,
    target: np.ndarray,
) -> float
```

Compute Learned Perceptual Image Patch Similarity (LPIPS) using the AlexNet backbone. Images are normalized to [-1, 1] internally.

Requires `lpips` and `torch` packages. Returns `float("nan")` if they are not installed.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pred` | `np.ndarray` | Predicted BGR image (uint8) |
| `target` | `np.ndarray` | Target BGR image (uint8) |

**Returns:** LPIPS distance as `float` (lower is better, more perceptually similar).

```python
from landmarkdiff.evaluation import compute_lpips
lpips_score = compute_lpips(result["output"], target_image)
```

---

### `compute_fid`

```python
def compute_fid(
    real_dir: str,
    generated_dir: str,
) -> float
```

Compute Frechet Inception Distance (FID) between two directories of images. Uses torch-fidelity for GPU-accelerated computation. Requires at least 50 images in each directory for statistically meaningful results.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_dir` | `str` | Path to directory of real/target images |
| `generated_dir` | `str` | Path to directory of generated/predicted images |

**Returns:** FID score as `float` (lower is better, more realistic distribution).

**Raises:** `ImportError` if `torch-fidelity` is not installed.

```python
from landmarkdiff.evaluation import compute_fid
fid = compute_fid("data/real_faces/", "output/predictions/")
```

---

### `compute_identity_similarity`

```python
def compute_identity_similarity(
    pred: np.ndarray,
    target: np.ndarray,
) -> float
```

Compute ArcFace identity cosine similarity between two face images. Uses InsightFace's `buffalo_l` model (512-dimensional embeddings). Falls back to SSIM-based proxy if InsightFace is not available.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pred` | `np.ndarray` | Predicted face image (BGR, uint8) |
| `target` | `np.ndarray` | Target face image (BGR, uint8) |

**Returns:** Cosine similarity in range [0, 1] where 1 = identical identity.

```python
from landmarkdiff.evaluation import compute_identity_similarity
sim = compute_identity_similarity(result["output"], original_image)
print(f"Identity preserved: {sim > 0.6}")
```

---

### `classify_fitzpatrick_ita`

```python
def classify_fitzpatrick_ita(image: np.ndarray) -> str
```

Classify Fitzpatrick skin type from an image using the Individual Typology Angle (ITA). Samples from the central face region in LAB color space.

ITA formula: `arctan((L - 50) / b) * (180 / pi)`

**ITA thresholds** (Chardon et al. 1991):

| ITA range | Type | Description |
|-----------|------|-------------|
| > 55 | `"I"` | Very light |
| 41 to 55 | `"II"` | Light |
| 28 to 41 | `"III"` | Intermediate |
| 10 to 28 | `"IV"` | Tan |
| -30 to 10 | `"V"` | Brown |
| <= -30 | `"VI"` | Dark |

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` | BGR face image (uint8) |

**Returns:** Fitzpatrick type string: `"I"`, `"II"`, `"III"`, `"IV"`, `"V"`, or `"VI"`.

```python
from landmarkdiff.evaluation import classify_fitzpatrick_ita
skin_type = classify_fitzpatrick_ita(face_image)
print(f"Fitzpatrick type: {skin_type}")
```

---

### `evaluate_batch`

```python
def evaluate_batch(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    pred_landmarks: list[np.ndarray] | None = None,
    target_landmarks: list[np.ndarray] | None = None,
    procedures: list[str] | None = None,
    compute_identity: bool = False,
) -> EvalMetrics
```

Evaluate a batch of predicted vs. target images. Computes all metrics and automatically stratifies results by Fitzpatrick skin type and by procedure.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `list[np.ndarray]` | (required) | List of predicted BGR images |
| `targets` | `list[np.ndarray]` | (required) | List of target BGR images |
| `pred_landmarks` | `list[np.ndarray] \| None` | `None` | Predicted landmarks as `(N, 2)` arrays (required for NME) |
| `target_landmarks` | `list[np.ndarray] \| None` | `None` | Target landmarks as `(N, 2)` arrays (required for NME) |
| `procedures` | `list[str] \| None` | `None` | Procedure names for per-procedure breakdown |
| `compute_identity` | `bool` | `False` | Compute ArcFace identity similarity (slower, requires InsightFace) |

**Returns:** `EvalMetrics` with all computed values and stratifications.

```python
from landmarkdiff.evaluation import evaluate_batch

metrics = evaluate_batch(
    predictions=pred_images,
    targets=target_images,
    pred_landmarks=pred_lm_list,
    target_landmarks=target_lm_list,
    procedures=["rhinoplasty"] * len(pred_images),
    compute_identity=True,
)
print(metrics.summary())
```

---

## See Also

- [inference](inference.md): pipeline that generates predictions to evaluate
- [landmarks](landmarks.md): landmark extraction for NME computation
- [clinical](clinical.md): clinical flags that affect predictions
