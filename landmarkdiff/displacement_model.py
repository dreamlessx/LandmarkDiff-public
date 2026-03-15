"""Data-driven surgical displacement extraction and modeling.

Extracts real landmark displacements from before/after surgery image pairs,
classifies procedures based on regional displacement patterns, and fits
per-procedure statistical models that can replace the hand-tuned RBF
displacement vectors in ``manipulation.py``.

Typical usage::

    from landmarkdiff.displacement_model import (
        extract_displacements,
        extract_from_directory,
        DisplacementModel,
    )

    # Single pair
    result = extract_displacements(before_img, after_img)

    # Batch from directory
    all_displacements = extract_from_directory("data/surgery_pairs/")

    # Fit model
    model = DisplacementModel()
    model.fit(all_displacements)
    model.save("displacement_model.npz")

    # Generate displacement field
    field = model.get_displacement_field("rhinoplasty", intensity=0.7)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS

logger = logging.getLogger(__name__)

# Number of MediaPipe Face Mesh landmarks (468 face + 10 iris)
NUM_LANDMARKS = 478

# All supported procedures
PROCEDURES = list(PROCEDURE_LANDMARKS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalized_coords_2d(face: FaceLandmarks) -> np.ndarray:
    """Extract (478, 2) normalized [0, 1] coordinates from a FaceLandmarks object.

    ``FaceLandmarks.landmarks`` is (478, 3) with (x, y, z) in normalized space.
    We take only the x, y columns.
    """
    return face.landmarks[:, :2].copy()


def _compute_alignment_quality(
    landmarks_before: np.ndarray,
    landmarks_after: np.ndarray,
) -> float:
    """Estimate alignment quality between two landmark sets.

    Uses a Procrustes-style analysis on landmarks that should *not* move during
    surgery (forehead, temples, ears) to measure how well the faces are aligned.
    A score of 1.0 means perfect alignment; lower values indicate pose/scale
    mismatches that contaminate the displacement signal.

    Args:
        landmarks_before: (478, 2) normalized coordinates.
        landmarks_after: (478, 2) normalized coordinates.

    Returns:
        Quality score in [0, 1].
    """
    # Stable landmarks: forehead, temple region, outer face oval
    # These should exhibit near-zero displacement after surgery.
    stable_indices = [
        10,
        109,
        67,
        103,
        54,
        21,
        162,
        127,  # left forehead/temple
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,  # right forehead/temple
        234,
        93,  # outer cheek anchors
    ]
    stable_indices = [i for i in stable_indices if i < NUM_LANDMARKS]

    before_stable = landmarks_before[stable_indices]
    after_stable = landmarks_after[stable_indices]

    # RMS displacement on stable points
    diffs = after_stable - before_stable
    rms = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))

    # Map RMS to quality: 0 displacement -> 1.0, rms >= 0.05 (5% of image) -> 0.0
    quality = float(np.clip(1.0 - rms / 0.05, 0.0, 1.0))
    return quality


# ---------------------------------------------------------------------------
# Procedure classification
# ---------------------------------------------------------------------------


def classify_procedure(displacements: np.ndarray) -> str:
    """Classify which surgical procedure was performed from displacement vectors.

    Computes the mean displacement magnitude within each procedure's landmark
    region (as defined by ``PROCEDURE_LANDMARKS``) and returns the procedure
    with the highest regional activity.

    Args:
        displacements: (478, 2) displacement vectors (after - before) in
            normalized coordinate space.

    Returns:
        Procedure name string, one of ``PROCEDURES``, or ``"unknown"`` if
        no region shows significant displacement.
    """
    magnitudes = np.linalg.norm(displacements, axis=1)

    best_procedure = "unknown"
    best_score = 0.0

    for procedure, indices in PROCEDURE_LANDMARKS.items():
        valid_indices = [i for i in indices if i < len(magnitudes)]
        if not valid_indices:
            continue

        region_mag = magnitudes[valid_indices]
        # Use mean magnitude in the region as the score
        score = float(np.mean(region_mag))

        if score > best_score:
            best_score = score
            best_procedure = procedure

    # If the best score is negligible, classify as unknown
    # Threshold: mean displacement < 0.002 (~1 pixel at 512x512)
    if best_score < 0.002:
        logger.debug(
            "No significant displacement detected (best=%.5f). Classified as 'unknown'.",
            best_score,
        )
        return "unknown"

    return best_procedure


# ---------------------------------------------------------------------------
# Single-pair extraction
# ---------------------------------------------------------------------------


def extract_displacements(
    before_img: np.ndarray,
    after_img: np.ndarray,
    min_detection_confidence: float = 0.5,
) -> dict | None:
    """Extract landmark displacements from a before/after surgery image pair.

    Runs MediaPipe Face Mesh on both images, computes per-landmark
    displacement vectors, classifies the procedure, and evaluates
    alignment quality.

    Args:
        before_img: Pre-surgery BGR image as numpy array.
        after_img: Post-surgery BGR image as numpy array.
        min_detection_confidence: Minimum face detection confidence for
            MediaPipe (default 0.5).

    Returns:
        Dictionary with keys:
            - ``landmarks_before``: (478, 2) normalized coordinates
            - ``landmarks_after``: (478, 2) normalized coordinates
            - ``displacements``: (478, 2) displacement vectors
            - ``magnitude``: (478,) per-landmark displacement magnitudes
            - ``procedure``: classified procedure name or ``"unknown"``
            - ``quality_score``: float in [0, 1] indicating alignment quality

        Returns ``None`` if face detection fails on either image.
    """
    # Extract landmarks from both images
    face_before = extract_landmarks(before_img, min_detection_confidence=min_detection_confidence)
    if face_before is None:
        logger.warning("Face detection failed on before image.")
        return None

    face_after = extract_landmarks(after_img, min_detection_confidence=min_detection_confidence)
    if face_after is None:
        logger.warning("Face detection failed on after image.")
        return None

    # Get normalized 2D coordinates
    coords_before = _normalized_coords_2d(face_before)
    coords_after = _normalized_coords_2d(face_after)

    # Compute displacements
    displacements = coords_after - coords_before
    magnitudes = np.linalg.norm(displacements, axis=1)

    # Classify procedure
    procedure = classify_procedure(displacements)

    # Evaluate alignment quality
    quality = _compute_alignment_quality(coords_before, coords_after)

    return {
        "landmarks_before": coords_before,
        "landmarks_after": coords_after,
        "displacements": displacements,
        "magnitude": magnitudes,
        "procedure": procedure,
        "quality_score": quality,
    }


# ---------------------------------------------------------------------------
# Batch extraction from directory
# ---------------------------------------------------------------------------


def extract_from_directory(
    pairs_dir: str | Path,
    min_detection_confidence: float = 0.5,
    min_quality: float = 0.0,
) -> list[dict]:
    """Batch-extract displacements from a directory of before/after image pairs.

    Supports two naming conventions:
        - ``<name>_before.{png,jpg,...}`` / ``<name>_after.{png,jpg,...}``
        - ``<name>_input.{png,jpg,...}`` / ``<name>_target.{png,jpg,...}``

    Args:
        pairs_dir: Path to directory containing image pairs.
        min_detection_confidence: Passed to ``extract_displacements``.
        min_quality: Minimum alignment quality score to include a pair
            in the results (default 0.0 = include all).

    Returns:
        List of displacement dictionaries (same format as
        ``extract_displacements``), each augmented with:
            - ``pair_name``: stem of the pair (e.g. ``"patient_001"``)
            - ``before_path``: path to the before image
            - ``after_path``: path to the after image
    """
    pairs_dir = Path(pairs_dir)
    if not pairs_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pairs_dir}")

    # Collect all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    all_files = {
        f.stem.lower(): f
        for f in pairs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    }

    # Find pairs using both naming conventions
    pairs: list[tuple[str, Path, Path]] = []
    seen_stems: set[str] = set()

    for stem_lower, filepath in all_files.items():
        # Convention 1: *_before / *_after
        for before_suffix, after_suffix in [("_before", "_after"), ("_input", "_target")]:
            if stem_lower.endswith(before_suffix):
                base = stem_lower[: -len(before_suffix)]
                after_stem = base + after_suffix
                if after_stem in all_files and base not in seen_stems:
                    # Use original-case paths
                    before_path = filepath
                    after_path = all_files[after_stem]
                    pairs.append((base, before_path, after_path))
                    seen_stems.add(base)

    if not pairs:
        logger.warning("No image pairs found in %s", pairs_dir)
        return []

    logger.info("Found %d image pairs in %s", len(pairs), pairs_dir)

    results: list[dict] = []
    for pair_name, before_path, after_path in sorted(pairs):
        logger.info("Processing pair: %s", pair_name)

        # Load images
        before_img = cv2.imread(str(before_path))
        if before_img is None:
            logger.warning("Failed to load before image: %s", before_path)
            continue

        after_img = cv2.imread(str(after_path))
        if after_img is None:
            logger.warning("Failed to load after image: %s", after_path)
            continue

        # Extract displacements
        result = extract_displacements(
            before_img, after_img, min_detection_confidence=min_detection_confidence
        )
        if result is None:
            logger.warning("Skipping pair %s: face detection failed.", pair_name)
            continue

        # Filter by quality
        if result["quality_score"] < min_quality:
            logger.info(
                "Skipping pair %s: quality %.3f < threshold %.3f",
                pair_name,
                result["quality_score"],
                min_quality,
            )
            continue

        # Augment with metadata
        result["pair_name"] = pair_name
        result["before_path"] = str(before_path)
        result["after_path"] = str(after_path)
        results.append(result)

    logger.info(
        "Successfully extracted %d / %d pairs (%.0f%%)",
        len(results),
        len(pairs),
        100.0 * len(results) / max(len(pairs), 1),
    )
    return results


# ---------------------------------------------------------------------------
# Displacement model
# ---------------------------------------------------------------------------


class DisplacementModel:
    """Statistical model of per-procedure surgical displacements.

    Aggregates displacement vectors from multiple before/after pairs and
    computes per-procedure, per-landmark statistics (mean, std, min, max).
    Can then generate displacement fields for use in the conditioning
    pipeline, replacing hand-tuned RBF vectors.

    Attributes:
        procedures: List of procedure names the model has data for.
        stats: Nested dict ``{procedure: {stat_name: array}}``.
        n_samples: Dict ``{procedure: int}`` sample counts.
    """

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, np.ndarray]] = {}
        self.n_samples: dict[str, int] = {}
        self._fitted = False

    @property
    def procedures(self) -> list[str]:
        """Return list of procedures the model has been fitted on."""
        return list(self.stats.keys())

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    def fit(self, displacement_list: list[dict]) -> None:
        """Fit the model from a list of extracted displacement dictionaries.

        Groups displacements by classified procedure and computes per-landmark
        statistics for each group.

        Args:
            displacement_list: List of dicts as returned by
                ``extract_displacements`` or ``extract_from_directory``.
                Each must contain ``"displacements"`` (478, 2) and
                ``"procedure"`` (str) keys.

        Raises:
            ValueError: If ``displacement_list`` is empty or contains no
                valid displacement data.
        """
        if not displacement_list:
            raise ValueError("displacement_list is empty.")

        # Group by procedure
        procedure_groups: dict[str, list[np.ndarray]] = {}
        for entry in displacement_list:
            proc = entry.get("procedure", "unknown")
            disp = entry.get("displacements")
            if disp is None:
                logger.warning("Skipping entry without 'displacements' key.")
                continue
            if disp.shape != (NUM_LANDMARKS, 2):
                logger.warning(
                    "Skipping entry with unexpected shape %s (expected (%d, 2)).",
                    disp.shape,
                    NUM_LANDMARKS,
                )
                continue

            if proc not in procedure_groups:
                procedure_groups[proc] = []
            procedure_groups[proc].append(disp)

        if not procedure_groups:
            raise ValueError("No valid displacement data found in displacement_list.")

        # Compute per-procedure statistics
        self.stats = {}
        self.n_samples = {}

        for proc, disp_arrays in procedure_groups.items():
            stacked = np.stack(disp_arrays, axis=0)  # (N, 478, 2)
            n = stacked.shape[0]

            self.stats[proc] = {
                "mean": np.mean(stacked, axis=0),  # (478, 2)
                "std": np.std(stacked, axis=0),  # (478, 2)
                "min": np.min(stacked, axis=0),  # (478, 2)
                "max": np.max(stacked, axis=0),  # (478, 2)
                "median": np.median(stacked, axis=0),  # (478, 2)
                "mean_magnitude": np.mean(  # (478,)
                    np.linalg.norm(stacked, axis=2), axis=0
                ),
            }
            self.n_samples[proc] = n
            logger.info(
                "Fitted procedure '%s': %d samples, mean magnitude=%.5f",
                proc,
                n,
                float(np.mean(self.stats[proc]["mean_magnitude"])),
            )

        self._fitted = True

    def get_displacement_field(
        self,
        procedure: str,
        intensity: float = 1.0,
        noise_scale: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a displacement field for a given procedure and intensity.

        Returns the mean displacement scaled by ``intensity``, optionally
        with Gaussian noise added (scaled by per-landmark std).

        Args:
            procedure: Procedure name (must exist in the fitted model).
            intensity: Scaling factor for the mean displacement. 1.0 = average
                observed displacement; 0.5 = half intensity; etc.
            noise_scale: If > 0, adds Gaussian noise with this many standard
                deviations of variation. 0.0 = deterministic mean field.
            rng: NumPy random generator for reproducible noise. If ``None``
                and ``noise_scale > 0``, uses ``np.random.default_rng()``.

        Returns:
            (478, 2) displacement field in normalized coordinate space.

        Raises:
            RuntimeError: If the model has not been fitted.
            KeyError: If the procedure is not in the model.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if procedure not in self.stats:
            available = ", ".join(self.procedures)
            raise KeyError(f"Procedure '{procedure}' not in model. Available: {available}")

        proc_stats = self.stats[procedure]
        field = proc_stats["mean"].copy() * intensity

        if noise_scale > 0:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(
                loc=0.0,
                scale=proc_stats["std"] * noise_scale,
            )
            field += noise

        return field.astype(np.float32)

    def get_summary(self, procedure: str | None = None) -> dict:
        """Get a human-readable summary of the model statistics.

        Args:
            procedure: If provided, return summary for one procedure.
                If ``None``, return summaries for all procedures.

        Returns:
            Dictionary with summary statistics.
        """
        if not self._fitted:
            return {"fitted": False}

        procs = [procedure] if procedure else self.procedures
        summary = {"fitted": True, "procedures": {}}

        for proc in procs:
            if proc not in self.stats:
                continue
            s = self.stats[proc]
            summary["procedures"][proc] = {
                "n_samples": self.n_samples[proc],
                "global_mean_magnitude": float(np.mean(s["mean_magnitude"])),
                "global_max_magnitude": float(np.max(s["mean_magnitude"])),
                "top_landmarks": _top_k_landmarks(s["mean_magnitude"], k=10),
            }

        return summary

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk as a ``.npz`` file.

        The file contains:
            - Per-procedure stat arrays keyed as ``{procedure}__{stat_name}``
            - A JSON metadata string with sample counts and procedure list

        Args:
            path: Output file path. Extension ``.npz`` is added if missing.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        arrays: dict[str, np.ndarray] = {}
        for proc, proc_stats in self.stats.items():
            for stat_name, arr in proc_stats.items():
                key = f"{proc}__{stat_name}"
                arrays[key] = arr

        # Store metadata as a JSON string encoded to bytes
        metadata = {
            "procedures": self.procedures,
            "n_samples": self.n_samples,
            "num_landmarks": NUM_LANDMARKS,
        }
        arrays["__metadata__"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)

        np.savez_compressed(str(path), **arrays)
        logger.info("Saved displacement model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> DisplacementModel:
        """Load a fitted model from a ``.npz`` file.

        Supports two formats:
        1. ``save()`` format: keys like ``{proc}__{stat}`` with ``__metadata__``
        2. ``extract_displacements.py`` format: keys like ``{proc}_{stat}``
           with a ``procedures`` array

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            A fitted ``DisplacementModel`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(str(path), allow_pickle=False)
        model = cls()

        # Format 1: save() format with __metadata__
        if "__metadata__" in data.files:
            meta_bytes = data["__metadata__"].tobytes()
            metadata = json.loads(meta_bytes.decode("utf-8"))
            model.n_samples = {k: int(v) for k, v in metadata["n_samples"].items()}

            for proc in metadata["procedures"]:
                model.stats[proc] = {}
                for key in data.files:
                    if key.startswith(f"{proc}__"):
                        stat_name = key[len(f"{proc}__") :]
                        model.stats[proc][stat_name] = data[key]

        # Format 2: extract_displacements.py format with procedures array
        elif "procedures" in data.files:
            procedures = [str(p) for p in data["procedures"]]
            # Map from extraction script key names to DisplacementModel stat names
            stat_map = {
                "mean": "mean",
                "std": "std",
                "median": "median",
                "min": "min",
                "max": "max",
                "mag_mean": "mean_magnitude",
                "mag_std": "std_magnitude",
                "count": "_count",
            }
            for proc in procedures:
                model.stats[proc] = {}
                for ext_key, model_key in stat_map.items():
                    npz_key = f"{proc}_{ext_key}"
                    if npz_key in data.files:
                        arr = data[npz_key]
                        if model_key == "_count":
                            model.n_samples[proc] = int(arr)
                        else:
                            model.stats[proc][model_key] = arr

                # Ensure count is set
                if proc not in model.n_samples:
                    model.n_samples[proc] = 0

        else:
            raise ValueError(f"Unrecognized displacement model format. Keys: {data.files[:10]}")

        # Validate loaded model is not empty
        if not model.stats:
            raise ValueError(
                f"Displacement model at {path} contains no procedure data. "
                f"File may be corrupted or empty. Keys found: {data.files[:10]}"
            )
        for proc, stats in model.stats.items():
            if not stats:
                raise ValueError(
                    f"Displacement model at {path} has no statistics for "
                    f"procedure '{proc}'. File may be corrupted."
                )

        model._fitted = True
        logger.info(
            "Loaded displacement model from %s (%d procedures, %s samples)",
            path,
            len(model.procedures),
            model.n_samples,
        )
        return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _top_k_landmarks(
    magnitudes: np.ndarray,
    k: int = 10,
) -> list[dict]:
    """Return the top-k landmarks by mean displacement magnitude.

    Args:
        magnitudes: (478,) array of per-landmark magnitudes.
        k: Number of top landmarks to return.

    Returns:
        List of dicts with ``index`` and ``magnitude`` keys, sorted
        descending by magnitude.
    """
    top_indices = np.argsort(magnitudes)[::-1][:k]
    return [{"index": int(idx), "magnitude": float(magnitudes[idx])} for idx in top_indices]


def visualize_displacements(
    before_img: np.ndarray,
    result: dict,
    scale: float = 10.0,
    arrow_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw displacement arrows on the before image for visual inspection.

    Args:
        before_img: BGR image (will be copied).
        result: Displacement dict from ``extract_displacements``.
        scale: Arrow length multiplier (displacements are small in
            normalized space, so scale up for visibility).
        arrow_color: BGR color for arrows.
        thickness: Arrow line thickness.

    Returns:
        Annotated BGR image.
    """
    canvas = before_img.copy()
    h, w = canvas.shape[:2]

    coords_before = result["landmarks_before"]
    displacements = result["displacements"]

    for i in range(NUM_LANDMARKS):
        bx = int(coords_before[i, 0] * w)
        by = int(coords_before[i, 1] * h)
        dx = int(displacements[i, 0] * w * scale)
        dy = int(displacements[i, 1] * h * scale)

        # Only draw if displacement is above noise floor
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1.0:
            continue

        cv2.arrowedLine(
            canvas,
            (bx, by),
            (bx + dx, by + dy),
            arrow_color,
            thickness,
            tipLength=0.3,
        )

    # Add procedure label and quality score
    proc = result.get("procedure", "unknown")
    quality = result.get("quality_score", 0.0)
    label = f"{proc} (quality={quality:.2f})"
    cv2.putText(
        canvas,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return canvas
