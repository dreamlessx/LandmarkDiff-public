"""TPS warping for synthetic pair generation.

Only warps deformable tissue - rigid structures (teeth, sclera) get
rigid translation instead. Prevents "rubber teeth" from naive TPS.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_tps_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> cv2.ThinPlateSplineShapeTransformer:
    """Fit a TPS transform from src to dst points."""
    src = src_pts.reshape(1, -1, 2).astype(np.float32)
    dst = dst_pts.reshape(1, -1, 2).astype(np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]

    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(dst, src, matches)
    return tps


def _subsample_control_points(
    src: np.ndarray,
    dst: np.ndarray,
    max_points: int = 80,
    anchor_stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all displaced points + sparse anchors. ~80 pts instead of 478, ~30x faster."""
    displacements = np.linalg.norm(dst - src, axis=1)
    displaced_mask = displacements > 0.5  # moved by > 0.5px
    displaced_idx = np.where(displaced_mask)[0]

    # Add sparse anchors from non-displaced landmarks
    non_displaced_idx = np.where(~displaced_mask)[0]
    anchor_idx = non_displaced_idx[::anchor_stride]

    selected = np.concatenate([displaced_idx, anchor_idx])

    # If still too many, subsample anchors more aggressively
    if len(selected) > max_points:
        n_anchors = max_points - len(displaced_idx)
        if n_anchors > 0:
            step = max(1, len(non_displaced_idx) // n_anchors)
            anchor_idx = non_displaced_idx[::step][:n_anchors]
            selected = np.concatenate([displaced_idx, anchor_idx])
        else:
            selected = displaced_idx[:max_points]

    selected = np.unique(selected)
    return src[selected], dst[selected]


def warp_image_tps(
    image: np.ndarray,
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    rigid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply TPS warp to an image with optional rigid region preservation."""
    h, w = image.shape[:2]

    src_pts = src_landmarks.astype(np.float32)
    dst_pts = dst_landmarks.astype(np.float32)

    # Subsample control points for speed (478 -> ~80)
    src_sub, dst_sub = _subsample_control_points(src_pts, dst_pts)

    # Compute TPS coefficients on subsampled points
    map_x, map_y = _compute_tps_map(src_sub, dst_sub, w, h)

    # Warp the image
    warped = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if rigid_mask is not None:
        # For rigid regions, compute mean translation and apply rigidly
        rigid_translation = _compute_rigid_translation(src_pts, dst_pts, rigid_mask, w, h)
        rigid_warped = _apply_rigid_translation(image, rigid_translation)

        # Composite: use rigid warp in rigid regions, TPS elsewhere
        mask_f = rigid_mask.astype(np.float32)
        if len(mask_f.shape) == 2:
            mask_f = np.stack([mask_f] * 3, axis=-1)
        mask_f = mask_f / 255.0 if mask_f.max() > 1 else mask_f
        warped = (rigid_warped * mask_f + warped * (1 - mask_f)).astype(np.uint8)

    return warped


def _compute_tps_map(
    src: np.ndarray,
    dst: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build remap arrays from TPS control points via RBF interpolation."""
    # Displacement at control points
    dx = dst[:, 0] - src[:, 0]
    dy = dst[:, 1] - src[:, 1]

    # Create grid
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_x = grid_x.astype(np.float64)
    grid_y = grid_y.astype(np.float64)

    # RBF interpolation using TPS kernel: r^2 * log(r)
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    n = len(src)
    if n == 0:
        return map_x, map_y

    # Solve TPS system for x and y displacements
    weights_x = _solve_tps_weights(src, dx)
    weights_y = _solve_tps_weights(src, dy)

    # Evaluate on grid (vectorized for speed)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    pts = np.stack([flat_x, flat_y], axis=1)

    disp_x = _evaluate_tps(pts, src, weights_x)
    disp_y = _evaluate_tps(pts, src, weights_y)

    map_x = (flat_x - disp_x).reshape(height, width)
    map_y = (flat_y - disp_y).reshape(height, width)

    return map_x, map_y


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    """TPS radial basis function: r^2 * log(r), with r=0 -> 0."""
    result = np.zeros_like(r)
    mask = r > 0
    result[mask] = r[mask] ** 2 * np.log(r[mask])
    return result


def _solve_tps_weights(
    control_pts: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Solve TPS system -> weight vector [w1..wn, a0, a1, a2]."""
    n = len(control_pts)

    # Build kernel matrix K (vectorized)
    diff = control_pts[:, np.newaxis, :] - control_pts[np.newaxis, :, :]  # (n, n, 2)
    r_mat = np.sqrt((diff**2).sum(axis=2))  # (n, n)
    K = np.zeros((n, n))
    nz = r_mat > 0
    K[nz] = r_mat[nz] ** 2 * np.log(r_mat[nz])

    # Build system matrix [K P; P^T 0]
    P = np.hstack([np.ones((n, 1)), control_pts])  # (n, 3)

    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    # Regularization for numerical stability
    L[:n, :n] += np.eye(n) * 1e-6

    rhs = np.zeros(n + 3)
    rhs[:n] = values

    try:
        weights = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(L, rhs, rcond=None)[0]

    return weights


def _evaluate_tps(
    points: np.ndarray,
    control_pts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate TPS at arbitrary points (vectorized)."""
    n = len(control_pts)
    w = weights[:n]
    a = weights[n:]  # affine: a0 + a1*x + a2*y

    # Affine component
    result = a[0] + a[1] * points[:, 0] + a[2] * points[:, 1]

    # Vectorized RBF evaluation in batches to limit memory
    batch_size = 50000
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        batch = points[start:end]  # (M, 2)

        # Compute all distances at once: (M, n)
        dx = batch[:, 0:1] - control_pts[:, 0]  # (M, n) via broadcasting
        dy = batch[:, 1:2] - control_pts[:, 1]  # (M, n)
        r = np.sqrt(dx**2 + dy**2)

        # TPS kernel: r^2 * log(r), with r=0 -> 0
        kernel = np.zeros_like(r)
        mask = r > 0
        kernel[mask] = r[mask] ** 2 * np.log(r[mask])

        # Weighted sum across all control points
        result[start:end] += kernel @ w

    return result


def _compute_rigid_translation(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Compute mean translation for rigid regions."""
    # Find control points inside rigid mask
    inside = []
    for i, (x, y) in enumerate(src):
        ix, iy = int(x), int(y)
        if 0 <= ix < width and 0 <= iy < height and mask[iy, ix] > 0:
            inside.append(i)

    if not inside:
        return np.array([0.0, 0.0])

    dx = np.mean(dst[inside, 0] - src[inside, 0])
    dy = np.mean(dst[inside, 1] - src[inside, 1])
    return np.array([dx, dy])


def _apply_rigid_translation(
    image: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply rigid translation to an image."""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def generate_random_warp(
    landmarks: np.ndarray,
    procedure_indices: list[int],
    max_displacement: float = 15.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate randomly warped landmarks for synthetic data."""
    rng = rng or np.random.default_rng()
    result = landmarks.copy()

    for idx in procedure_indices:
        if idx < len(landmarks):
            dx = rng.uniform(-max_displacement, max_displacement)
            dy = rng.uniform(-max_displacement, max_displacement)
            result[idx, 0] += dx
            result[idx, 1] += dy

    return result
