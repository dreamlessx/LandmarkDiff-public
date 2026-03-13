# landmarkdiff.evaluation

Evaluation metrics with Fitzpatrick skin type stratification.

## Classes

### `EvalMetrics`

Container for all evaluation metrics with Fitzpatrick stratification.

**Attributes:**
- `fid` (float | None): Frechet Inception Distance
- `lpips_scores` (list[float]): Per-image LPIPS scores
- `ssim_scores` (list[float]): Per-image SSIM scores
- `nme_scores` (list[float]): Per-image NME scores
- `identity_sim_scores` (list[float]): Per-image identity similarity

**Methods:**
- `to_dict() -> dict`: Export all metrics as JSON-serializable dict

## Functions

### `compute_nme(predicted, target) -> float`

Normalized mean error between predicted and target landmarks (normalized by inter-ocular distance).

### `compute_ssim(image1, image2) -> float`

Structural similarity index between two images. Uses 11x11 Gaussian window.

### `compute_lpips(image1, image2, net="alex") -> float`

Learned perceptual image patch similarity. Lower is better.

### `compute_fid(real_dir, fake_dir) -> float`

Frechet Inception Distance between two image directories. Requires 50+ images.

### `classify_fitzpatrick_ita(image, landmarks) -> str`

Classify skin type (I-VI) using the Individual Typology Angle from forehead region.

### `evaluate_batch(predictions, targets, landmarks_pred, landmarks_target) -> EvalMetrics`

Compute all metrics for a batch of predictions.
